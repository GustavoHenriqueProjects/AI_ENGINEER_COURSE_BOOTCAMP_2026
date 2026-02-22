"""Transcrição de áudio com Whisper."""

import os
import tempfile

_whisper_model = None


def _get_whisper_model():
    """Carrega o modelo Whisper uma vez e reutiliza (cache em memória)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


def transcrever_audio(audio_bytes: bytes) -> str:
    """Transcreve áudio em WAV para texto em português."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(path, language="pt")
        return " ".join(s.text.strip() for s in segments).strip() or "(áudio não reconhecido)"
    finally:
        os.unlink(path)
