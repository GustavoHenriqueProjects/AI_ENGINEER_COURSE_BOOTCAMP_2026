from typing import Any, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str
    messages: list
    stream: Optional[bool] = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class RagRequest(BaseModel):
    query: str
    # Filtro opcional: chaves iguais às gravadas no Chroma (ver RAG_METADATA_KEY_LANG em config).
    # pt → tipo, secao, tema, pais, cidade, ano | en → type, section, theme, country, city, year
    # Ex.: {"secao": "ENDERECOS"} — só blocos cujo cabeçalho tenha esses valores
    metadata_filter: Optional[dict[str, str]] = None
    # None = usar RAG_K em api/config.py
    top_k: Optional[int] = None
    model: Optional[str] = "llama3.2:1b"
    # None = usar RAG_TEMPERATURE em api/config.py
    temperature: Optional[float] = None
    # None = sem limite na geração (Ollama sem num_predict); inteiro = limite opcional por pedido
    max_tokens: Optional[int] = None
    # True = só documentos da empresa quando não há contexto; False = banco opcional + conhecimento geral
    use_rag: bool = True
    messages: Optional[list[Any]] = None
