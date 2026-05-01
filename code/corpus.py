"""Corpus loader + chunker.

Walks data/{hackerrank,claude,visa}/ and produces a flat list of CorpusChunk
objects ready for indexing. Each markdown file is split by `##` headings with
a soft token cap, so retrieved chunks stay under ~350 tokens (well below
LLM context limits when 5 of them are stuffed into a prompt).

We cache the parsed chunks to disk so subsequent runs are instant + deterministic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

import yaml

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    DATA_DIR,
    INDEX_CACHE_DIR,
)
from schema import CorpusChunk
from taxonomy import product_area_for

CACHE_FILE = INDEX_CACHE_DIR / "chunks.json"

# Approximate token = 0.75 words. We don't need exact tokenization for chunking.
WORDS_PER_TOKEN = 0.75


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) / WORDS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body. Returns (meta, body). Tolerant of
    files that have no frontmatter (which exist in the corpus)."""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, m.group(2)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

# Split on level-2 headings; if a section is still too long we hard-wrap it.
_H2_RE = re.compile(r"\n(?=##\s+)")


def _hard_wrap(section: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    words = section.split()
    if not words:
        return []
    max_words = int(max_tokens / WORDS_PER_TOKEN)  # words per chunk
    overlap_words = int(overlap_tokens / WORDS_PER_TOKEN)
    if len(words) <= max_words:
        return [section]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _chunk_body(body: str) -> List[str]:
    """Split a markdown body into retrieval-sized chunks."""
    body = body.strip()
    if not body:
        return []
    # First split on H2 sections — keeps semantic boundaries.
    sections = [s.strip() for s in _H2_RE.split(body) if s.strip()]
    out: List[str] = []
    for sec in sections:
        if _approx_tokens(sec) <= CHUNK_TARGET_TOKENS:
            out.append(sec)
        else:
            out.extend(_hard_wrap(sec, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS))
    return out


# ---------------------------------------------------------------------------
# Corpus walk
# ---------------------------------------------------------------------------


def _iter_company_files(company: str) -> Iterable[Path]:
    company_dir = DATA_DIR / company
    if not company_dir.exists():
        return []
    return sorted(company_dir.rglob("*.md"))


# Product-area resolution lives in taxonomy.py for auditability.


def load_chunks(use_cache: bool = True) -> List[CorpusChunk]:
    """Load every chunk in the corpus. Cached to disk after first run."""
    if use_cache and CACHE_FILE.exists():
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return [CorpusChunk(**c) for c in payload]

    chunks: List[CorpusChunk] = []
    for company in ("hackerrank", "claude", "visa"):
        for path in _iter_company_files(company):
            rel = path.relative_to(DATA_DIR)
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta, body = _parse_frontmatter(raw)
            title = (meta.get("title") or path.stem).strip()
            breadcrumbs = list(meta.get("breadcrumbs") or [])
            source_url = (meta.get("source_url") or "").strip()
            product_area = product_area_for(company, rel.as_posix())

            for i, text in enumerate(_chunk_body(body)):
                chunk_id = f"{company}/{rel.as_posix()}#chunk-{i}"
                chunks.append(
                    CorpusChunk(
                        chunk_id=chunk_id,
                        company=company,
                        product_area=product_area,
                        title=title,
                        breadcrumbs=breadcrumbs,
                        source_url=source_url,
                        text=text,
                        rel_path=rel.as_posix(),
                    )
                )

    # Cache
    if use_cache:
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                [c.__dict__ for c in chunks],
                f,
                ensure_ascii=False,
                indent=None,
            )

    return chunks


def product_areas_for_company(company: str, chunks: List[CorpusChunk]) -> List[str]:
    """Return the canonical product_area vocabulary for a given company,
    derived from the loaded chunks (so it always matches what the agent
    can actually emit)."""
    return sorted({c.product_area for c in chunks if c.company == company})


if __name__ == "__main__":  # pragma: no cover
    import sys

    chunks = load_chunks(use_cache=False)
    print(f"Loaded {len(chunks)} chunks", file=sys.stderr)
    for company in ("hackerrank", "claude", "visa"):
        n = sum(1 for c in chunks if c.company == company)
        print(f"  {company}: {n} chunks", file=sys.stderr)
        print(
            f"  {company} areas: {product_areas_for_company(company, chunks)}",
            file=sys.stderr,
        )
