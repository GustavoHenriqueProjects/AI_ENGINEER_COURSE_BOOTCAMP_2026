#!/usr/bin/env python3
"""
Recria o índice Chroma a partir do manual configurado em `RAG_KNOWLEDGE_FILE`.

Ative o ambiente virtual na raiz do projeto:

    source chatbot_pdf/bin/activate

Depois:

    python build_rag.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def main() -> int:
    from api.rag_index import build_chroma_index

    try:
        n = build_chroma_index()
        print(f"OK: {n} documentos indexados.")
        return 0
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
