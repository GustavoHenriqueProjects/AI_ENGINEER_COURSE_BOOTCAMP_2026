from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx
from fastapi.responses import StreamingResponse

from api.config import (
    GEMINI_API_BASE,
    GEMINI_CLOUD_PAYLOAD_MODEL,
    GEMINI_MODEL,
    OPENAI_API_BASE,
)
from api.schemas import ChatRequest

logger = logging.getLogger(__name__)


def _gemini_error_text_for_user(status_code: int, response: httpx.Response) -> str:
    """Mensagem para o cliente; não logar a resposta inteira em nível ERROR (pode conter dados sensíveis)."""
    msg = ""
    try:
        data = response.json()
        err = data.get("error")
        if isinstance(err, dict):
            msg = (err.get("message") or "").strip()
        elif err:
            msg = str(err).strip()
    except Exception:
        raw = (response.text or "").strip()
        if raw:
            msg = raw[:500]
    if status_code == 429:
        lines = [
            "Limite da API Google atingido (HTTP 429 — Too Many Requests).",
            "Aguarde alguns minutos ou reduza o ritmo de chamadas.",
            "Confira cota e uso em Google AI Studio ou Google Cloud Console.",
        ]
        if msg:
            lines.append(f"Detalhe: {msg}")
        return "\n".join(lines) + "\n"
    return (
        f"Erro na API Gemini (HTTP {status_code}).\n"
        + (f"{msg}\n" if msg else "")
    )


async def _gemini_load_body(response: httpx.Response) -> None:
    try:
        await response.aread()
    except Exception:
        pass


def stream_static_text(text: str) -> StreamingResponse:
    return StreamingResponse(iter([text]), media_type="text/plain")


def _build_ollama_payload(request: ChatRequest) -> dict:
    options = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens

    return {
        "model": request.model,
        "messages": request.messages,
        "stream": request.stream,
        "options": options,
    }


async def get_chat_completion(request: ChatRequest, ollama_url: str) -> str:
    """Executa chamada sem stream e retorna o texto completo."""
    payload = _build_ollama_payload(
        ChatRequest(
            model=request.model,
            messages=request.messages,
            stream=False,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


def stream_chat_response(
    request: ChatRequest,
    ollama_url: str,
    *,
    stream_prefix: str = "",
) -> StreamingResponse:
    """Se stream_prefix estiver preenchido, envia antes do stream."""

    async def generate():
        if stream_prefix:
            yield stream_prefix

        ollama_payload = _build_ollama_payload(request)

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Usando stream=True no httpx para lidar com a resposta linha a linha
            async with client.stream("POST", ollama_url, json=ollama_payload) as response:
                response.raise_for_status()

                if request.stream:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            # O Ollama retorna o conteúdo em ['message']['content']
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            
                            # Opcional: verificar se o stream terminou
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    # Caso stream seja False, lemos a resposta inteira de uma vez
                    full_response = await response.aread()
                    data = json.loads(full_response)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    return StreamingResponse(generate(), media_type="text/plain")


def routes_to_gemini(model: str) -> bool:
    """True se a requisição deve ir para a API Google (inclui o marcador do front)."""
    m = model.strip()
    if m == GEMINI_CLOUD_PAYLOAD_MODEL:
        return True
    return m.lower().startswith("gemini")


def is_openai_cloud_model(model: str) -> bool:
    m = model.strip().lower()
    return (
        m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o2")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("chatgpt-")
        or (m.startswith("ft:") and "gpt" in m)
    )


def _messages_to_gemini_rest(messages: list) -> tuple[Optional[str], list[dict[str, Any]]]:
    """systemInstruction + contents no formato REST do Google."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "")
        text = m.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        if role == "system":
            if text.strip():
                system_parts.append(text)
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role in ("assistant", "model"):
            contents.append({"role": "model", "parts": [{"text": text}]})
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _messages_openai(messages: list) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role", "")
        if role not in ("system", "user", "assistant"):
            continue
        text = m.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        out.append({"role": role, "content": text})
    return out


def _gemini_generation_config(request: ChatRequest) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if request.temperature is not None:
        cfg["temperature"] = request.temperature
    if request.max_tokens is not None:
        cfg["maxOutputTokens"] = request.max_tokens
    return cfg


def stream_gemini_http(
    request: ChatRequest,
    api_key: str,
    *,
    stream_prefix: str = "",
) -> StreamingResponse:
    system_instruction, contents = _messages_to_gemini_rest(request.messages)
    if not contents:
        return stream_static_text(
            "Erro: nenhuma mensagem de usuário para enviar ao Gemini.\n"
        )

    body: dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    gen_cfg = _gemini_generation_config(request)
    if gen_cfg:
        body["generationConfig"] = gen_cfg

    stream_url = f"{GEMINI_API_BASE}/models/{request.model}:streamGenerateContent"
    once_url = f"{GEMINI_API_BASE}/models/{request.model}:generateContent"

    async def generate():
        if stream_prefix:
            yield stream_prefix
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if not request.stream:
                    r = await client.post(
                        once_url,
                        params={"key": api_key},
                        json=body,
                    )
                    try:
                        r.raise_for_status()
                    except httpx.HTTPStatusError:
                        logger.warning(
                            "[Gemini] HTTP %s (generateContent)",
                            r.status_code,
                        )
                        yield _gemini_error_text_for_user(r.status_code, r)
                        return
                    obj = r.json()
                    err = obj.get("error")
                    if err:
                        msg = err.get("message", err) if isinstance(err, dict) else err
                        yield f"\n[Erro Gemini: {msg}]\n"
                        return
                    for c in obj.get("candidates", []):
                        for part in (c.get("content") or {}).get("parts", []):
                            t = part.get("text")
                            if t:
                                yield t
                    return

                params = {"key": api_key, "alt": "sse"}
                async with client.stream(
                    "POST", stream_url, params=params, json=body
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError:
                        await _gemini_load_body(response)
                        logger.warning(
                            "[Gemini] HTTP %s (streamGenerateContent)",
                            response.status_code,
                        )
                        yield _gemini_error_text_for_user(
                            response.status_code, response
                        )
                        return
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        err = obj.get("error")
                        if err:
                            msg = err.get("message", err) if isinstance(err, dict) else err
                            yield f"\n[Erro Gemini: {msg}]\n"
                            break
                        for c in obj.get("candidates", []):
                            for part in (c.get("content") or {}).get("parts", []):
                                t = part.get("text")
                                if t:
                                    yield t
        except httpx.HTTPStatusError as e:
            await _gemini_load_body(e.response)
            logger.warning("[Gemini] HTTP %s", e.response.status_code)
            yield _gemini_error_text_for_user(e.response.status_code, e.response)
        except Exception as e:
            logger.exception("[Gemini] Erro inesperado")
            yield f"\n[Erro Gemini: {e}]\n"

    return StreamingResponse(generate(), media_type="text/plain")


def stream_openai_http(
    request: ChatRequest,
    api_key: str,
    *,
    stream_prefix: str = "",
) -> StreamingResponse:
    oa_messages = _messages_openai(request.messages)
    if not oa_messages:
        return stream_static_text(
            "Erro: nenhuma mensagem válida para a API OpenAI.\n"
        )

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": oa_messages,
        "stream": bool(request.stream),
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async def generate():
        if stream_prefix:
            yield stream_prefix
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if not request.stream:
                    r = await client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    obj = r.json()
                    err = obj.get("error")
                    if err:
                        msg = err.get("message", err) if isinstance(err, dict) else err
                        yield f"\n[Erro OpenAI: {msg}]\n"
                        return
                    choices = obj.get("choices") or []
                    if choices:
                        msg_content = (choices[0].get("message") or {}).get("content")
                        if msg_content:
                            yield msg_content
                    return

                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        err = obj.get("error")
                        if err:
                            msg = err.get("message", err) if isinstance(err, dict) else err
                            yield f"\n[Erro OpenAI: {msg}]\n"
                            break
                        for choice in obj.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if content:
                                yield content
        except httpx.HTTPStatusError as e:
            logger.warning("[OpenAI] HTTP %s", e.response.status_code)
            yield f"\n[Erro OpenAI HTTP: {e.response.status_code}]\n"
        except Exception as e:
            logger.exception("[OpenAI] Erro inesperado")
            yield f"\n[Erro OpenAI: {e}]\n"

    return StreamingResponse(generate(), media_type="text/plain")


def stream_llm_response(
    request: ChatRequest,
    ollama_url: str,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    *,
    stream_prefix: str = "",
) -> StreamingResponse:
    """
    Gemini (marcador ou gemini-*) → GOOGLE_API_KEY; modelo efetivo sempre GEMINI_MODEL.
    gpt-/o* → OPENAI_API_KEY | resto → Ollama.
    """
    if routes_to_gemini(request.model):
        if not google_api_key:
            return stream_static_text(
                "Erro: Gemini exige GOOGLE_API_KEY ou GEMINI_API_KEY no .env do backend.\n"
            )
        gemini_req = request.model_copy(update={"model": GEMINI_MODEL})
        return stream_gemini_http(
            gemini_req,
            google_api_key,
            stream_prefix=stream_prefix,
        )
    if is_openai_cloud_model(request.model):
        if not openai_api_key:
            return stream_static_text(
                "Erro: modelo OpenAI exige OPENAI_API_KEY no .env do backend.\n"
            )
        return stream_openai_http(
            request,
            openai_api_key,
            stream_prefix=stream_prefix,
        )
    return stream_chat_response(
        request,
        ollama_url,
        stream_prefix=stream_prefix,
    )
