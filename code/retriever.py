"""BM25 retriever scoped per company.

Why BM25 and not embeddings:
  * Tiny corpus (~774 docs / a few thousand chunks). BM25 is well-calibrated
    at this scale.
  * Tickets share vocabulary with docs ("test invite", "subscription",
    "lost card"). Lexical overlap is strong.
  * Fully deterministic, no model weights, no API call — keeps the agent
    reproducible (a scoring criterion) and adds zero runtime dependencies
    beyond rank_bm25.
  * Failure mode is explicit: low top-1 score => corpus does not cover
    this question => escalate. That's exactly the safety behavior the
    rubric asks for.

We keep a *per-company* index plus a *global* index. The pipeline picks the
right scope based on the ticket's company (with a fallback to global when
the company is unknown / "None").
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from config import RETRIEVAL_TOP_K
from schema import CorpusChunk, RetrievalResult

# A small custom stopword list. Avoids pulling all of NLTK at runtime.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "as", "from",
    "have", "has", "had", "i", "you", "he", "she", "we", "they", "my",
    "your", "our", "their", "me", "him", "her", "us", "them", "do", "does",
    "did", "not", "no", "so", "than", "then", "there", "here", "what",
    "which", "who", "how", "when", "where", "why", "can", "could", "would",
    "should", "may", "might", "will", "shall", "about", "up", "down",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric tokens + stopword removal.

    Deterministic. No stemming — the corpus and tickets share enough surface
    forms that stemming actively hurts (e.g. "tests" and "testing" mean
    different things in HackerRank).
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _enrich(chunk: CorpusChunk) -> str:
    """Concatenate fields with title/breadcrumb upweighting via repetition.

    This is the BM25-equivalent of a heading boost: repeating high-signal
    tokens makes them count more in the bag-of-words score.
    """
    title = chunk.title or ""
    crumbs = " ".join(chunk.breadcrumbs or [])
    # Title repeated ×3, breadcrumbs ×2, body ×1
    return f"{title} {title} {title} {crumbs} {crumbs} {chunk.text}"


class CompanyIndex:
    """One BM25 index for a subset of chunks (e.g. all HackerRank chunks)."""

    def __init__(self, chunks: List[CorpusChunk]):
        self.chunks = chunks
        if not chunks:
            self._bm25 = None
            self._docs: List[List[str]] = []
            return
        self._docs = [_tokenize(_enrich(c)) for c in chunks]
        self._bm25 = BM25Okapi(self._docs)

    def search(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        if not self._bm25 or not self._docs:
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        # Sort by score desc, take top-k. Stable for identical scores.
        ranked = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i),
        )[:k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


class Retriever:
    """Holds per-company indexes plus a global index."""

    def __init__(self, all_chunks: List[CorpusChunk]):
        by_company: Dict[str, List[CorpusChunk]] = {}
        for c in all_chunks:
            by_company.setdefault(c.company, []).append(c)
        self._indexes: Dict[str, CompanyIndex] = {
            company: CompanyIndex(chunks) for company, chunks in by_company.items()
        }
        self._global = CompanyIndex(all_chunks)

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        k: int = RETRIEVAL_TOP_K,
    ) -> List[RetrievalResult]:
        """If company is known, search only its index. Otherwise global."""
        if company and company in self._indexes:
            return self._indexes[company].search(query, k=k)
        return self._global.search(query, k=k)

    def search_global(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        return self._global.search(query, k=k)


def build_query(issue: str, subject: str) -> str:
    """Combine subject + issue. Subject often has the cleanest topic phrase."""
    parts = [p for p in (subject, issue) if p and p.strip()]
    return " ".join(parts)
