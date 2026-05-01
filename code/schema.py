"""Lightweight dataclasses for the agent pipeline.

Keeping these as plain dataclasses (not pydantic) avoids a heavy dep and
keeps the JSON contract obvious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Ticket:
    """One row from support_tickets.csv."""

    issue: str
    subject: str
    company: str  # "HackerRank" | "Claude" | "Visa" | "None" | other (will be normalized)
    row_index: int = 0  # for stable logging

    @property
    def normalized_company(self) -> str:
        """Map noisy `company` field into one of {hackerrank, claude, visa, unknown}."""
        c = (self.company or "").strip().lower()
        if c in {"hackerrank", "hr"}:
            return "hackerrank"
        if c == "claude":
            return "claude"
        if c == "visa":
            return "visa"
        return "unknown"


@dataclass
class CorpusChunk:
    """A retrieved or indexed chunk of corpus markdown."""

    chunk_id: str  # stable: "<company>/<rel_path>#chunk-<idx>"
    company: str  # "hackerrank" | "claude" | "visa"
    product_area: str  # top-level subfolder under company (e.g. "screen", "privacy")
    title: str
    breadcrumbs: List[str] = field(default_factory=list)
    source_url: str = ""
    text: str = ""
    rel_path: str = ""

    def label(self) -> str:
        """Human-friendly label used in prompts and justifications."""
        crumb = " > ".join(self.breadcrumbs) if self.breadcrumbs else self.product_area
        return f"[{self.company}/{crumb}] {self.title}"


@dataclass
class RetrievalResult:
    chunk: CorpusChunk
    score: float


@dataclass
class TriageDecision:
    """Output of the deterministic rule layer that runs *before* the LLM."""

    short_circuit: bool  # if True, skip retrieval + LLM and use canned response
    status: Optional[str] = None  # "replied" | "escalated" — only set if short_circuit
    request_type: Optional[str] = None
    response: Optional[str] = None
    product_area: Optional[str] = None
    justification: str = ""
    rule_name: str = ""  # which rule fired (for debugging)


@dataclass
class AgentOutput:
    """Final per-row result, written to output.csv."""

    issue: str
    subject: str
    company: str
    response: str
    product_area: str
    status: str  # replied | escalated
    request_type: str  # product_issue | feature_request | bug | invalid
    justification: str

    def as_csv_row(self) -> dict:
        return {
            "issue": self.issue,
            "subject": self.subject,
            "company": self.company,
            "response": self.response,
            "product_area": self.product_area,
            "status": self.status,
            "request_type": self.request_type,
            "justification": self.justification,
        }
