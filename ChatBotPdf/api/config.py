"""Configuração central do backend (URLs, modelo de embedding, textos da assistente)."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Ollama (chat / geração — embeddings do RAG usam EMBEDDING_MODEL + sentence-transformers).
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")


def _first_non_empty_env(*names: str) -> Optional[str]:
    for name in names:
        v = os.getenv(name)
        if v and v.strip():
            return v.strip()
    return None


# OpenAI (ChatGPT / gpt-* / o1-* …) — só é usada quando o modelo for da OpenAI.
OPENAI_API_KEY = _first_non_empty_env("OPENAI_API_KEY")

# Google Gemini — chave do Google AI; aceita GEMINI_API_KEY como alias.
GOOGLE_API_KEY = _first_non_empty_env("GOOGLE_API_KEY", "GEMINI_API_KEY")

# Modelo Gemini usado na API quando há GOOGLE_API_KEY (o front não escolhe a variante).
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip() or "gemini-2.0-flash"

# Valor que o cliente envia em `model` para significar “usar Gemini via servidor”.
GEMINI_CLOUD_PAYLOAD_MODEL = os.getenv(
    "GEMINI_CLOUD_PAYLOAD_MODEL",
    "__gemini_google__",
).strip() or "__gemini_google__"

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

# RAG / Chroma
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "db")
# Sentence-transformers: indexação Chroma no /rag. Chat continua no Ollama.
# Padrão: BAAI/bge-m3 (multilingual, open source); bom equilíbrio qualidade / RAM moderada (~16 GB).
# Pouca RAM/CPU: intfloat/multilingual-e5-small ou multilingual-e5-base —
# `api/rag_index.py` aplica query:/passage: só para nomes E5.
# Ao trocar o modelo de embedding, apague CHROMA_PERSIST_DIR e rode: python build_rag.py
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

# Base de conhecimento (blocos com cabeçalho TIPO= | SECAO= | …); caminho rel. ao CWD ou à raiz do projeto
RAG_KNOWLEDGE_FILE = os.getenv("RAG_KNOWLEDGE_FILE", "txt/manual_brain4care.txt")
# Chaves gravadas no Chroma: `pt` → tipo, secao, tema, pais, cidade, ano | `en` → type, section, theme, country, city, year
# Ao mudar, apague CHROMA_PERSIST_DIR e rode python build_rag.py. O metadata_filter na API deve usar as mesmas chaves.
_RAG_META_LANG = (os.getenv("RAG_METADATA_KEY_LANG") or "en").strip().lower()
RAG_METADATA_KEY_LANG = _RAG_META_LANG if _RAG_META_LANG in ("pt", "en") else "en"

# Chunkamento legado (não usado no indexador de blocos estruturados; mantido para scripts/compat)
RAG_CHAR_CHUNK_SIZE = int(os.getenv("RAG_CHAR_CHUNK_SIZE", "550"))
RAG_CHAR_CHUNK_OVERLAP = int(os.getenv("RAG_CHAR_CHUNK_OVERLAP", "120"))
# Retriever: "similarity" = só os trechos mais parecidos com a pergunta (recomendado).
# "mmr" = mistura relevância com diversidade (pode trazer trechos pouco relacionados).
RAG_RETRIEVER_SEARCH_TYPE = (os.getenv("RAG_RETRIEVER_SEARCH_TYPE") or "similarity").strip().lower()
# Top-k na busca por similaridade (o body /rag também envia `top_k`).
RAG_SIMILARITY_K = int(os.getenv("RAG_SIMILARITY_K", "10"))
# MMR (só se RAG_RETRIEVER_SEARCH_TYPE=mmr): fetch_k >= k; lambda alto = mais foco na pergunta
RAG_MMR_K = int(os.getenv("RAG_MMR_K", "10"))
RAG_MMR_FETCH_K = int(os.getenv("RAG_MMR_FETCH_K", "32"))
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.88"))

# Recuperação: score menor = mais similar no Chroma (L2)
RELEVANCE_SCORE_MAX = float(os.getenv("RAG_RELEVANCE_SCORE_MAX", "1.35"))
SEARCH_K_MULTIPLIER = int(os.getenv("RAG_SEARCH_K_MULTIPLIER", "4"))
SEARCH_K_MIN = int(os.getenv("RAG_SEARCH_K_MIN", "12"))

# Identidade da assistente (rodapé opcional em respostas triviais / matemática)
ASSISTANT_NAME = "Fernanda"
BRAND_NAME = "Brain4care"

ASSISTANT_FOOTER = (
    f"Sou a {ASSISTANT_NAME}, assistente virtual da {BRAND_NAME}. "
    "Estou aqui para ajudar com informações sobre produtos e outras dúvidas relacionadas à empresa. "
    "Como posso te ajudar?"
)

# Quando a busca no Chroma não traz contexto útil — sem regex de “pergunta genérica”
NO_DOCUMENT_CONTEXT_REPLY = (
    "Não encontrei essa informação nos documentos indexados da empresa. "
    "Só posso ajudar com o que estiver na base da Brain4care. "
    "Se a dúvida for sobre outro assunto, posso tentar se você reformular com base nos materiais disponíveis."
)
