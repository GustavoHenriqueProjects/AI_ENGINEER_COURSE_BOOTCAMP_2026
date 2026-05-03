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
    RAG_MMR_FETCH_K,
    RAG_MMR_K,
    RAG_MMR_LAMBDA,
    RAG_RETRIEVER_SEARCH_TYPE,
    RAG_SIMILARITY_K,
)
from api.rag_index import chroma_index_exists, open_chroma_vectorstore
from api.schemas import ChatRequest, RagRequest
from api.services.chat_service import (
    stream_llm_response,
    stream_static_text,
)

logger = logging.getLogger(__name__)

# Mensagem do usuário: pergunta + contexto recuperado (placeholders explícitos).
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


def _merged_retrieval_context(docs: list) -> str:
    """Junta vários trechos recuperados, sem repetir o mesmo texto."""
    if not docs:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for d in docs:
        t = (d.page_content or "").strip()
        if t and t not in seen:
            seen.add(t)
            parts.append(t)
    return "\n\n---\n\n".join(parts)


def _build_chroma_retriever(db, k_similarity: int):
    """MMR maximiza diversidade entre trechos; para perguntas pontuais use similarity (padrão)."""
    mode = (RAG_RETRIEVER_SEARCH_TYPE or "similarity").strip().lower()
    k_sim = max(1, min(k_similarity, 40))
    if mode == "mmr":
        k_mmr = max(1, min(RAG_MMR_K, 40))
        fetch_k = max(RAG_MMR_FETCH_K, k_mmr)
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_mmr,
                "fetch_k": min(fetch_k, 60),
                "lambda_mult": RAG_MMR_LAMBDA,
            },
        )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_sim},
    )


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

        k_similarity = (
            int(request.top_k) if request.top_k is not None else RAG_SIMILARITY_K
        )
        k_similarity = max(1, min(k_similarity, 40))

        where = _chroma_where(request.metadata_filter)
        if where is not None:
            if (RAG_RETRIEVER_SEARCH_TYPE or "similarity").strip().lower() == "mmr":
                logger.warning(
                    "[RAG] metadata_filter ativo: usando busca por similaridade (sem MMR)."
                )
            retrieved = db.similarity_search(
                request.query,
                k=k_similarity,
                filter=where,
            )
        else:
            retriever = _build_chroma_retriever(db, k_similarity)
            retrieved = retriever.invoke(request.query)

        context_text = _merged_retrieval_context(retrieved)

        if not context_text.strip():
            context_text = (
                "(Nenhum trecho do manual foi recuperado para esta pergunta; "
                "responda de acordo com as regras quando não houver contexto.)"
            )

        system_prompt = (
            f"Você é a {ASSISTANT_NAME}, assistente virtual da {BRAND_NAME}. "
            "1. Saudações (olá, boa noite, etc) devem ser respondidas de forma amigável. "
            "2. Para dúvidas, use EXCLUSIVAMENTE o contexto fornecido, salvo saudação, "
            "conta matemática simples ou dúvida em que o conhecimento geral seja adequado. "
            "3. Se a pergunta não for saudação e não estiver no contexto, diga que não possui a informação. "
            "4. Quando o contexto trouxer várias unidades, cidades, datas ou itens relacionados à pergunta, "
            "cite todos de forma organizada, sem omitir referências presentes no texto."
        )

        user_content = RAG_USER_MESSAGE_TEMPLATE.format(
            question=request.query.strip(),
            context=context_text,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        llm_request = ChatRequest(
            model=request.model,
            messages=messages,
            stream=True,
            temperature=0.0 if request.use_rag else request.temperature,
            max_tokens=request.max_tokens,
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
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                ollama_url,
                openai_api_key,
                google_api_key,
            )
        except Exception as fb:
            logger.error("[RAG] Fallback falhou: %s", fb)
            return stream_static_text(f"Erro: {e}\n")
