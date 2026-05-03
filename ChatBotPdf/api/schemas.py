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
    # None = usar RAG_SIMILARITY_K na config do servidor
    top_k: Optional[int] = None
    model: Optional[str] = "llama3.2:1b"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 400
    # True = só documentos da empresa quando não há contexto; False = banco opcional + conhecimento geral
    use_rag: bool = True
    messages: Optional[list[Any]] = None
