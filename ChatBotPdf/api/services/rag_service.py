"""
Pipeline /rag simplificado:

1) Pré-processamento: saudação + matemática simples.
2) Busca no Chroma (sempre).
3) `use_rag=True`: responde somente com contexto do banco.
4) `use_rag=False`: usa contexto do banco quando existir + permite conhecimento geral.
"""

from __future__ import annotations

from typing import Any

import logging

from api.config import (
    ASSISTANT_NAME,
    BRAND_NAME,
    RAG_CONTEXT_MAX_CHARS,
    RAG_K,
    RAG_TEMPERATURE,
)
from api.rag_index import chroma_index_exists, open_chroma_vectorstore
from api.schemas import ChatRequest, RagRequest
from api.services.chat_service import (
    stream_llm_response,
    stream_static_text,
)

logger = logging.getLogger(__name__)

RAG_USER_MESSAGE_TEMPLATE = (
    "Essa é a pergunta do usuário:\n"
    "{question}\n\n"
    "Esse é o contexto:\n"
    "{context}"
)

_MISSING_INDEX_MSG = (
    "Erro: índice vetorial (Chroma) não encontrado. "
    "Ative o ambiente: source chatbot_pdf/bin/activate — "
    "depois, na raiz do projeto: python build_rag.py\n"
)


def _chroma_where(user_filter: dict[str, str] | None) -> dict[str, Any] | None:
    """Converte filtro da API em `where` do Chroma (igualdade em metadados indexados)."""
    if not user_filter:
        return None
    norm: dict[str, str] = {}
    for k, v in user_filter.items():
        if v is None or str(v).strip() == "":
            continue
        norm[k.strip().lower()] = str(v).strip()
    if not norm:
        return None
    if len(norm) == 1:
        k, v = next(iter(norm.items()))
        return {k: v}
    return {"$and": [{k: v} for k, v in norm.items()]}


def _retrieve_documents(db, query: str, k: int, where: dict[str, Any] | None) -> list:
    pairs = db.similarity_search_with_score(query, k=k, filter=where)
    return [doc for doc, _score in pairs]

async def handle_rag(
    request: RagRequest,
    ollama_url: str,
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
) -> object:
    try:
        if not chroma_index_exists():
            logger.error("[RAG] Índice Chroma ausente (rode python build_rag.py).")
            return stream_static_text(_MISSING_INDEX_MSG)

        db = open_chroma_vectorstore()

        k = int(request.top_k) if request.top_k is not None else RAG_K
        k = max(1, min(k, 40))
        where = _chroma_where(request.metadata_filter)

        retrieved = _retrieve_documents(db, request.query, k, where)

        chunks: list[str] = []
        for doc in retrieved:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            if RAG_CONTEXT_MAX_CHARS > 0:
                text = text[:RAG_CONTEXT_MAX_CHARS]
            chunks.append(text)
        context_text = "\n\n---\n\n".join(chunks)
        if not context_text.strip():
            context_text = (
                "(Nenhum trecho do manual foi recuperado para esta pergunta; "
                "responda de acordo com as regras quando não houver contexto.)"
            )

        _log_preview = (context_text[:400] + "…") if len(context_text) > 400 else context_text
        logger.info(
            "[RAG] retrieval query=%r k=%s n_docs=%s context_chars=%s "
            "(log só: primeiros 400 chars)=%r",
            request.query.strip(),
            k,
            len(retrieved),
            len(context_text),
            _log_preview,
        )

        system_prompt = (
            f"Você é a {ASSISTANT_NAME}, assistente virtual da {BRAND_NAME}. "
            "1. Saudações (olá, boa noite, etc.) devem ser respondidas de forma amigável. "
        )

        user_content = RAG_USER_MESSAGE_TEMPLATE.format(
            question=request.query.strip(),
            context=context_text,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        eff_temperature = (
            request.temperature if request.temperature is not None else RAG_TEMPERATURE
        )
        llm_request = ChatRequest(
            model=request.model,
            messages=messages,
            stream=True,
            temperature=eff_temperature,
        )

        logger.info(
            "[RAG] ollama model=%s temperature=%s (sem limite max_tokens / num_predict)",
            llm_request.model,
            llm_request.temperature,
        )

        return stream_llm_response(
            llm_request,
            ollama_url,
            openai_api_key,
            google_api_key,
            stream_prefix="",
        )

    except Exception as e:
        logger.error("[RAG] Erro: %s", e, exc_info=True)
        try:
            return stream_llm_response(
                ChatRequest(
                    model=request.model,
                    messages=[{"role": "user", "content": request.query}],
                    stream=True,
                    temperature=(
                        request.temperature
                        if request.temperature is not None
                        else RAG_TEMPERATURE
                    ),
                ),
                ollama_url,
                openai_api_key,
                google_api_key,
            )
        except Exception as fb:
            logger.error("[RAG] Fallback falhou: %s", fb)
            return stream_static_text(f"Erro: {e}\n")
