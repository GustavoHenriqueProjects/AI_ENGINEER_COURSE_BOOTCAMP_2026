"""Serviços do ChatBot (LLM, transcrição de áudio)."""

from .transcriber import transcrever_audio
from .llm import get_client, stream_chat, stream_chat_generator

__all__ = ["transcrever_audio", "get_client", "stream_chat", "stream_chat_generator"]
