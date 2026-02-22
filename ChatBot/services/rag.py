"""RAG: carrega PDFs da empresa, indexa em Chroma e fornece contexto para o chat."""

import os
import re
import sys
import unicodedata
from pathlib import Path


def _normalize_for_match(text: str) -> str:
    """Remove acentos e colapsa espaços para comparação robusta."""
    nfd = unicodedata.normalize("NFD", text.lower())
    sem_acentos = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return " ".join(sem_acentos.split())

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    RAG_CANDIDATES,
    RAG_DOCUMENT_METADATA,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)


def _get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")


def _get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_pdfs_from_data_dir():
    """Carrega todos os PDFs da pasta configurada em DATA_DIR, com metadados da empresa."""
    documents = []
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return documents
    for path in sorted(data_path.glob("**/*.pdf")):
        try:
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.update(RAG_DOCUMENT_METADATA)
            documents.extend(docs)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar {path}: {e}") from e
    return documents


def build_rag_index():
    """
    Lê os PDFs em DATA_DIR, divide em pedaços, gera embeddings e persiste no Chroma.
    Rode uma vez (ou quando adicionar novos PDFs): python -m ChatBot.build_rag
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_docs = load_pdfs_from_data_dir()
    if not raw_docs:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {DATA_DIR}. Coloque os PDFs da empresa nessa pasta e rode de novo."
        )
    splitter = _get_splitter()
    chunks = splitter.split_documents(raw_docs)
    embeddings = _get_embeddings()
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="empresa",
    )
    return len(chunks)


def _normalize_query_for_search(query: str) -> str:
    """Para perguntas sobre nomes, usa frase que bate com o PDF e melhora o recall."""
    q = query.strip().lower()
    if "diretoria executiva" in q and any(w in q for w in ("nomes", "pessoas", "quais", "composição", "composta", "integrantes")):
        return "A Diretoria Executiva é composta por"
    if "conselho" in q and any(w in q for w in ("nomes", "pessoas", "quais", "composição", "formado", "integrantes")):
        return "O Conselho de Administração é formado por"
    person = _extract_person_name(query)
    if person:
        return f"{person} Brain4care"
    return query.strip()


def _is_diretoria_question(search_query: str) -> bool:
    """Indica se a busca foi normalizada para Diretoria Executiva."""
    return search_query == "A Diretoria Executiva é composta por"


def _is_conselho_question(search_query: str) -> bool:
    """Indica se a busca foi normalizada para Conselho de Administração."""
    return search_query == "O Conselho de Administração é formado por"


def _chunk_is_diretoria_only(txt: str) -> bool:
    """True se o trecho contém Diretoria Executiva e a lista (não só Conselho)."""
    return "Diretoria Executiva" in txt and "composta por" in txt


def _chunk_has_conselho_list(txt: str) -> bool:
    """True se o trecho contém a lista do Conselho de Administração."""
    return "Conselho de Administração" in txt and "Marcos Bicudo" in txt


def _chunk_is_conselho_only(txt: str) -> bool:
    """True se o trecho é principalmente sobre Conselho (sem lista da Diretoria)."""
    return _chunk_has_conselho_list(txt) and not (
        "Diretoria Executiva" in txt and ("Plínio Targa" in txt or "Arnaldo Betta" in txt)
    )


def _chunk_is_diretoria_only_no_conselho(txt: str) -> bool:
    """True se o trecho tem só Diretoria, sem lista do Conselho."""
    return _chunk_is_diretoria_only(txt) and not _chunk_has_conselho_list(txt)


def _extract_year(query: str) -> str | None:
    """Extrai ano (19xx ou 20xx) da pergunta, se existir."""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", query)
    return m.group(1) if m else None


def _extract_person_name(query: str) -> str | None:
    """Extrai nome de pessoa em perguntas como 'Quem é Luis Pascoal?' ou 'Quem é o Plínio Targa?'."""
    q = query.strip()
    m = re.search(r"(?:quem\s+é|quem\s+foi)\s+(?:o\s+|a\s+)?([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{2,40}?)(?:\s*\?|$)", q, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if 3 <= len(name) <= 40 and not name.lower().startswith(("o ", "a ", "o que", "qual")):
            return name
    return None


def _build_extra_queries(query: str) -> list[str]:
    """Gera queries extras para busca múltipla e melhor recall."""
    extra = []
    year = _extract_year(query)
    if year:
        extra.append(f"{year} Brain4care")
    person = _extract_person_name(query)
    if person:
        extra.append(f"{person} Brain4care")
    extra.append(f"Brain4care {query.strip()[:80]}")
    return extra


def get_relevant_context(query: str, top_k: int = RAG_TOP_K) -> str:
    """
    Busca trechos relevantes no índice RAG (Chroma) com a pergunta do usuário e retorna um único texto.
    Deduplica e limita a top_k. Se o índice não existir ou estiver vazio, retorna "".
    """
    if not query or not query.strip():
        return ""
    chroma_path = os.path.abspath(CHROMA_DIR)
    if not os.path.isdir(chroma_path):
        print(f"[RAG] chroma_db não encontrada: {chroma_path}\nRode: cd ChatBot && python build_rag.py", file=sys.stderr)
        return ""
    try:
        embeddings = _get_embeddings()
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name="empresa",
        )
        k = min(RAG_CANDIDATES, 60)
        search_query = _normalize_query_for_search(query)
        person = _extract_person_name(query)
        docs = list(vectorstore.similarity_search(search_query, k=k))
        seen = {d.page_content.strip() for d in docs}
        for eq in _build_extra_queries(query):
            extra = vectorstore.similarity_search(eq, k=15)
            for d in extra:
                txt = d.page_content.strip()
                if txt and txt not in seen:
                    seen.add(txt)
                    docs.append(d)
        if person:
            person_norm = _normalize_for_match(person)
            docs = [d for d in docs if person_norm in _normalize_for_match(d.page_content)]
            if not docs:
                all_data = vectorstore._collection.get(include=["documents"])
                for txt in (all_data.get("documents") or []):
                    if txt and person_norm in _normalize_for_match(txt):
                        doc = type("Doc", (), {})()
                        doc.page_content = txt
                        docs.append(doc)
                if not docs:
                    print(f"[RAG] Nenhum trecho contém '{person}'. Índice em: {chroma_path}", file=sys.stderr)
        year = _extract_year(query)
        if year:
            docs_with_year = [d for d in docs if year in (d.page_content or "")]
            if not docs_with_year:
                all_data = vectorstore._collection.get(include=["documents"])
                for txt in (all_data.get("documents") or []):
                    if txt and year in txt:
                        doc = type("Doc", (), {})()
                        doc.page_content = txt
                        docs.append(doc)
            else:
                docs = docs_with_year
        unique_contents = []
        seen_list = set()
        is_diretoria = _is_diretoria_question(search_query)
        is_conselho = _is_conselho_question(search_query)
        for d in docs:
            txt = d.page_content.strip()
            if not txt or txt in seen_list:
                continue
            if is_diretoria:
                if _chunk_is_conselho_only(txt):
                    continue
                if not _chunk_is_diretoria_only(txt) and len(unique_contents) >= 5:
                    continue
            elif is_conselho:
                if _chunk_is_diretoria_only_no_conselho(txt):
                    continue
                if not _chunk_has_conselho_list(txt) and len(unique_contents) >= 5:
                    continue
            elif year and year not in txt:
                continue
            seen_list.add(txt)
            unique_contents.append(txt)
            if len(unique_contents) >= top_k:
                break
        if not unique_contents:
            print(f"[RAG] Nenhum trecho relevante. Índice em: {chroma_path}\nRode: cd ChatBot && python build_rag.py", file=sys.stderr)
            return ""
        return "\n\n---\n\n".join(unique_contents)
    except Exception as e:
        print(f"[RAG] Erro: {e}", file=sys.stderr)
        return ""
