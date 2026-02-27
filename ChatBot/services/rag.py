"""RAG: carrega PDFs da empresa, indexa em Chroma e fornece contexto para o chat."""

import os
import re
import sys
from pathlib import Path

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

# Cache: evita recriar embeddings e Chroma a cada pergunta (grande ganho de velocidade)
_embeddings_cache = None
_vectorstore_cache = None


def _get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
    return _embeddings_cache


def _get_vectorstore():
    """Vectorstore em cache: uma única conexão com Chroma por sessão."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        _vectorstore_cache = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings(),
            collection_name="empresa",
        )
    return _vectorstore_cache


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


def _is_governance_question(query: str) -> bool:
    """True se a pergunta é sobre composição da Diretoria ou Conselho."""
    q = query.strip().lower()
    return ("diretoria" in q and any(w in q for w in ("composta", "composição", "nomes", "quem", "quais", "compõe"))) or (
        "conselho" in q and any(w in q for w in ("formado", "composição", "nomes", "quem", "quais", "compõe", "administração"))
    )


def is_composition_question(query: str) -> bool:
    """True se a pergunta é sobre composição da Diretoria ou Conselho."""
    return _is_governance_question(query)


def is_person_question(query: str) -> bool:
    """True se a pergunta é 'Quem é [nome]?'."""
    return _extract_person_name(query) is not None


def is_year_question(query: str) -> bool:
    """True se a pergunta menciona um ano específico (ex: o que aconteceu em 2010?)."""
    return _extract_year(query) is not None


def _extract_person_name(query: str) -> str | None:
    """Extrai nome de pessoa em perguntas como 'Quem é Marcos Bicudo?' ou 'Quem é o Plínio Targa?'."""
    q = query.strip()
    m = re.search(r"(?:quem\s+é|quem\s+foi)\s+(?:o\s+|a\s+)?([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{2,50}?)(?:\s*\?|$)", q, re.IGNORECASE)
    if m:
        name = " ".join(m.group(1).strip().split())
        if 3 <= len(name) <= 50:
            return name
    return None


def _normalize_for_match(text: str) -> str:
    """Colapsa espaços para comparação robusta."""
    return " ".join((text or "").split()).lower()


def _extract_year(query: str) -> str | None:
    """Extrai ano (19xx ou 20xx) da pergunta, se existir."""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", query)
    return m.group(1) if m else None


def _has_governance_section(txt: str) -> bool:
    """True se o trecho contém Diretoria Executiva ou Conselho de Administração (normaliza espaços/caps)."""
    if not txt:
        return False
    t = " ".join(txt.split()).lower()
    return "diretoria executiva" in t or "conselho de administração" in t


def _merge_governance_chunks(chunks: list[str]) -> str:
    """
    Mescla chunks de governança cortados no meio da lista.
    Deduplica: mesma pessoa (antes do —) fica só a linha mais completa.
    """
    person_best = {}
    for txt in chunks:
        for line in txt.split("\n"):
            m = re.match(r"^(.+?)\s+—\s+(.+)$", line.strip())
            if m:
                person = " ".join(m.group(1).split())
                cargo = m.group(2).strip()
                full = f"{person} — {cargo}"
                if person not in person_best or len(cargo) > len(person_best[person].split(" — ", 1)[-1]):
                    person_best[person] = full
    result = []
    seen_h = set()
    seen_p = set()
    for txt in chunks:
        for line in txt.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(.+?)\s+—\s+", line)
            if m:
                person = " ".join(m.group(1).split())
                if person not in seen_p and person in person_best:
                    seen_p.add(person)
                    result.append(person_best[person])
            elif line not in seen_h:
                seen_h.add(line)
                result.append(line)
    return "\n".join(result)


def get_relevant_context(query: str, top_k: int = RAG_TOP_K) -> str:
    """
    Busca trechos relevantes no índice RAG (Chroma) com a pergunta do usuário e retorna um único texto.
    Usa vectorstore em cache e similarity_search (evita carregar a coleção inteira = muito mais rápido).
    """
    if not query or not query.strip():
        return ""
    if not os.path.isdir(CHROMA_DIR):
        return ""
    try:
        vectorstore = _get_vectorstore()
        q = query.strip()
        person = _extract_person_name(q)
        year = _extract_year(q)
        k = min(RAG_CANDIDATES, 30)

        if person:
            docs = vectorstore.similarity_search(person, k=k)
            person_norm = _normalize_for_match(person)
            person_chunks = [
                d.page_content.strip()
                for d in docs
                if d.page_content and person_norm in _normalize_for_match(d.page_content)
            ]
            if person_chunks:
                return "\n\n---\n\n".join(person_chunks[:top_k])

        if year:
            docs = vectorstore.similarity_search(q, k=k)
            year_chunks = [
                d.page_content.strip()
                for d in docs
                if d.page_content and year in d.page_content
            ]
            if year_chunks:
                return "\n\n---\n\n".join(year_chunks[:top_k])

        if _is_governance_question(q):
            docs = vectorstore.similarity_search(
                "diretoria executiva conselho de administração composição nomes cargos",
                k=k,
            )
            governance_chunks = [
                d.page_content.strip()
                for d in docs
                if d.page_content and _has_governance_section(d.page_content)
            ]
            if governance_chunks:
                return _merge_governance_chunks(governance_chunks)

        docs = list(vectorstore.similarity_search(q, k=k))
        unique_contents = []
        seen = set()
        for d in docs:
            txt = d.page_content.strip()
            if txt and txt not in seen:
                seen.add(txt)
                unique_contents.append(txt)
            if len(unique_contents) >= top_k:
                break
        if not unique_contents:
            return ""
        return "\n\n---\n\n".join(unique_contents)
    except Exception:
        return ""
