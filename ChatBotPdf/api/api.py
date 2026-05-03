import logging

from fastapi import FastAPI

from api.config import (
    GEMINI_CLOUD_PAYLOAD_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    OLLAMA_CHAT_URL,
)
from api.schemas import ChatRequest, RagRequest
from api.services.chat_service import stream_llm_response
from api.services.rag_service import handle_rag


logging.basicConfig(level=logging.INFO)
# Evita INFO do httpx com URL completa (Gemini usa ?key=… e vaza a chave nos logs).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI()


@app.get("/config")
def public_config():
    """Sinaliza capacidades do backend (sem expor chaves)."""
    return {
        "gemini_available": bool(GOOGLE_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "gemini_choice_value": GEMINI_CLOUD_PAYLOAD_MODEL,
        "openai_available": bool(OPENAI_API_KEY),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    return stream_llm_response(
        request, OLLAMA_CHAT_URL, OPENAI_API_KEY, GOOGLE_API_KEY
    )


@app.post("/rag")
async def rag(request: RagRequest):
    return await handle_rag(
        request, OLLAMA_CHAT_URL, OPENAI_API_KEY, GOOGLE_API_KEY
    )
