"""Configuração central do backend (URLs, chaves, RAG, identidade da assistente)."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _first_non_empty_env(*names: str) -> Optional[str]:
    for name in names:
        v = os.getenv(name)
        if v and v.strip():
            return v.strip()
    return None


OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

OPENAI_API_KEY = _first_non_empty_env("OPENAI_API_KEY")
GOOGLE_API_KEY = _first_non_empty_env("GOOGLE_API_KEY", "GEMINI_API_KEY")

GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip() or "gemini-2.0-flash"
GEMINI_CLOUD_PAYLOAD_MODEL = os.getenv(
    "GEMINI_CLOUD_PAYLOAD_MODEL",
    "__gemini_google__",
).strip() or "__gemini_google__"

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

RAG_KNOWLEDGE_FILE = os.getenv("RAG_KNOWLEDGE_FILE", "txt/manual_brain4care.txt")
_RAG_META_LANG = (os.getenv("RAG_METADATA_KEY_LANG") or "en").strip().lower()
RAG_METADATA_KEY_LANG = _RAG_META_LANG if _RAG_META_LANG in ("pt", "en") else "en"

# Recuperação /rag: igual a similarity_search_with_score(..., k=…) + truncagem do texto no prompt.
# Aceita também o env legado RAG_SIMILARITY_K se RAG_K não estiver definido.
_RAG_K_RAW = os.getenv("RAG_K") or os.getenv("RAG_SIMILARITY_K") or "4"
RAG_K = max(1, min(40, int(_RAG_K_RAW)))
# 0 = sem truncagem (todo o page_content de cada documento recuperado).
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS") or "0")
# Ollama no /rag: 0–0.3 = respostas mais fiéis ao contexto; ~0.2 é um bom padrão; 0.7+ = mais criativo.
_RAG_T = float(os.getenv("RAG_TEMPERATURE") or "0.2")
RAG_TEMPERATURE = max(0.0, min(2.0, _RAG_T))

ASSISTANT_NAME = "Fernanda"
BRAND_NAME = "Brain4care"
