"""Cliente LLM (OpenAI API compatível com Ollama)."""

from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, MODEL

OLLAMA_TIMEOUT = 120.0  # segundos — evita travamento se Ollama demorar

_client = None


def get_client() -> OpenAI:
    """Retorna o cliente OpenAI configurado para Ollama."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            timeout=OLLAMA_TIMEOUT,
        )
    return _client


def stream_chat(model: str, messages: list[dict]) -> str:
    """
    Envia mensagens ao modelo e retorna a resposta completa (acumulando o stream).
    Para exibir em tempo real no Streamlit, use st.write_stream(stream_chat_generator(...)).
    """
    client = get_client()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.1,
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


def stream_chat_generator(model: str, messages: list[dict]):
    """Generator para usar com st.write_stream()."""
    client = get_client()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=2048,  # Evita resposta truncada (ex.: lista completa da diretoria)
        temperature=0.1,  # Muito baixa = mais fiel ao contexto, evita inventar nomes
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
