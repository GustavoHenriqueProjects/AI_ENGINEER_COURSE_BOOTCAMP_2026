"""
Indexação Chroma a partir do manual estruturado (`RAG_KNOWLEDGE_FILE`).

O endpoint `/rag` só abre o banco persistido; para criar ou atualizar vetores,
rode na raiz do projeto: ``python build_rag.py`` (com o venv ativado).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from api.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RAG_KNOWLEDGE_FILE
from api.knowledge_loader import load_structured_knowledge_text

logger = logging.getLogger(__name__)

# Modelos E5 (intfloat) foram treinados com prefixos assimétricos em recuperação.
_E5_SUBSTRINGS = (
    "multilingual-e5-base",
    "multilingual-e5-small",
    "multilingual-e5-large",
)


def _embedding_uses_e5_prompts(model_name: str) -> bool:
    lower = model_name.lower().replace("\\", "/")
    return any(s in lower for s in _E5_SUBSTRINGS)


def huggingface_rag_embeddings() -> HuggingFaceEmbeddings:
    """Embeddings alinhados ao RAG: E5 usa `passage:` nos docs e `query:` na pergunta."""
    if _embedding_uses_e5_prompts(EMBEDDING_MODEL):
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "prompt": "passage: ",
                "normalize_embeddings": True,
            },
            query_encode_kwargs={
                "prompt": "query: ",
                "normalize_embeddings": True,
            },
        )
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_knowledge_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_file():
        return p
    root = _project_root()
    alt = root / rel
    if alt.is_file():
        return alt
    return p


def chroma_persist_path() -> Path:
    p = Path(CHROMA_PERSIST_DIR)
    return p.resolve() if p.is_absolute() else (_project_root() / p).resolve()


def resolve_chroma_persist_dir() -> str:
    """
    Caminho absoluto do SQLite do Chroma, ancorado na raiz do repositório.
    Evita CWD diferente (ex.: SQLITE_READONLY em pasta somente leitura).
    """
    out = chroma_persist_path()
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def chroma_index_exists() -> bool:
    """True se já existe persistência Chroma (após ``build_chroma_index``)."""
    return (chroma_persist_path() / "chroma.sqlite3").is_file()


def build_chroma_index() -> int:
    """
    Remove o diretório de persistência, reindexa o manual e grava no Chroma.

    Returns:
        Número de documentos indexados.

    Raises:
        FileNotFoundError: se `RAG_KNOWLEDGE_FILE` não existir.
        ValueError: se o manual não tiver blocos legíveis.
    """
    kb_path = resolve_knowledge_path(RAG_KNOWLEDGE_FILE)
    if not kb_path.is_file():
        raise FileNotFoundError(f"Arquivo de conhecimento não encontrado: {kb_path}")
    docs_chunks = load_structured_knowledge_text(kb_path)
    if not docs_chunks:
        raise ValueError(f"Nenhum bloco legível em {kb_path}")

    root = chroma_persist_path()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    embeddings = huggingface_rag_embeddings()
    Chroma.from_documents(
        documents=docs_chunks,
        embedding=embeddings,
        persist_directory=str(root),
    )
    logger.info("[RAG index] %s documentos em %s", len(docs_chunks), root)
    return len(docs_chunks)


def open_chroma_vectorstore() -> Chroma:
    """Abre o Chroma persistido (coleção padrão ``langchain``)."""
    persist = resolve_chroma_persist_dir()
    embeddings = huggingface_rag_embeddings()
    return Chroma(
        persist_directory=persist,
        embedding_function=embeddings,
    )
