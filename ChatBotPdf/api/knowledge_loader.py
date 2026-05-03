"""
Carrega o manual em blocos: primeira linha = cabeçalho KEY=VAL | KEY=VAL; metadados no Chroma.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

from api.config import RAG_METADATA_KEY_LANG


def _normalize_meta_key(raw_key: str) -> str:
    """
    Chaves estáveis para Chroma/filtros (ASCII): PAÍS e país viram `pais`,
    evitando ambiguidade com «pais» (família) e mantendo `metadata_filter` previsível.
    """
    k = raw_key.strip().lower()
    nk = unicodedata.normalize("NFD", k)
    return "".join(c for c in nk if unicodedata.category(c) != "Mn")


# Sinónimos no cabeçalho (PT/EN) → chave única no Chroma, conforme RAG_METADATA_KEY_LANG.
_META_KEY_ALIASES_PT: dict[str, str] = {
    "tipo": "tipo",
    "type": "tipo",
    "secao": "secao",
    "section": "secao",
    "tema": "tema",
    "theme": "tema",
    "pais": "pais",
    "country": "pais",
    "cidade": "cidade",
    "city": "cidade",
    "ano": "ano",
    "year": "ano",
}
_META_KEY_ALIASES_EN: dict[str, str] = {
    "tipo": "type",
    "type": "type",
    "secao": "section",
    "section": "section",
    "tema": "theme",
    "theme": "theme",
    "pais": "country",
    "country": "country",
    "cidade": "city",
    "city": "city",
    "ano": "year",
    "year": "year",
}


def _canonical_meta_key(norm_key: str) -> str:
    table = _META_KEY_ALIASES_EN if RAG_METADATA_KEY_LANG == "en" else _META_KEY_ALIASES_PT
    return table.get(norm_key, norm_key)


def _parse_header_line(first_line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in first_line.split("|"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = _canonical_meta_key(_normalize_meta_key(k))
        v = v.strip()
        if k and v is not None:
            out[k] = v
    return out


def load_structured_knowledge_text(path: str | Path) -> list[Document]:
    """
    Cada bloco (separado por linha em branco) vira um Document com:
    - page_content = bloco completo
    - metadata: chaves definidas por `RAG_METADATA_KEY_LANG` em `api/config.py` (`pt` ou `en`).
      `pt`: tipo, secao, tema, pais, cidade, ano. `en`: type, section, theme, country, city, year.
      Cabeçalhos TIPO/TYPE, PAÍS/COUNTRY, etc. são aceites em ambos os manuais; o loader unifica.
    """
    p = Path(path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    raw_blocks = re.split(r"\n\s*\n", text)
    out: list[Document] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if "=" not in first:
            out.append(
                Document(
                    page_content=block,
                    metadata={"formato": "bloco_sem_cabecalho_estruturado"},
                )
            )
            continue
        meta = _parse_header_line(first)
        if not meta:
            out.append(
                Document(
                    page_content=block,
                    metadata={"formato": "cabecalho_ilegivel"},
                )
            )
            continue
        out.append(Document(page_content=block, metadata=meta))
    return out
