"""Serviços do ChatBot (LLM, transcrição de áudio, RAG)."""

from .transcriber import transcrever_audio
from .llm import get_client, stream_chat, stream_chat_generator
from .rag import get_relevant_context, build_rag_index

__all__ = [
    "transcrever_audio",
    "get_client",
    "stream_chat",
    "stream_chat_generator",
    "get_relevant_context",
    "build_rag_index",
]
