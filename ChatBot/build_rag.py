"""
Script para indexar os PDFs da empresa no Chroma (RAG).
Rode uma vez após colocar os PDFs em ChatBot/data/pdf/ e quando adicionar novos arquivos.

  cd ChatBot
  python build_rag.py

Requisito: Ollama com o modelo de embeddings instalado: ollama pull nomic-embed-text
"""

import sys
from pathlib import Path

# Garante que o diretório ChatBot está no path (para imports)
_chatbot_dir = Path(__file__).resolve().parent
if str(_chatbot_dir) not in sys.path:
    sys.path.insert(0, str(_chatbot_dir))

from services.rag import build_rag_index
from config import DATA_DIR


def main():
    print(f"Procurando PDFs em: {DATA_DIR}")
    try:
        n = build_rag_index()
        print(f"Índice RAG criado com sucesso. {n} trechos indexados.")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao construir índice: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
