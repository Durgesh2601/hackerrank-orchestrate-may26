"""Centralized configuration. All knobs live here so the agent stays deterministic
and reproducible. Read from environment variables, never hardcode secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. Idempotent.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATA_DIR = REPO_ROOT / "data"
TICKETS_DIR = REPO_ROOT / "support_tickets"
INDEX_CACHE_DIR = DATA_DIR / "index"  # gitignored
LOG_DIR = CODE_DIR / "logs"  # local debug logs, gitignored via __pycache__-adjacent path

LOG_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "900"))
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "60"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

# Minimum BM25 score for the top-1 result before we consider the corpus to
# have answered the question. Tuned empirically against sample tickets.
# Below this, we escalate or reply with "out of scope".
RETRIEVAL_MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "4.0"))

# Score gap between top-1 and top-2 below which we treat retrieval as
# ambiguous (multiple equally-relevant docs => caution).
RETRIEVAL_AMBIGUITY_GAP = float(os.environ.get("RETRIEVAL_AMBIGUITY_GAP", "0.5"))

# Chunking
CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]

ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

# Canned responses — used by the rule layer for short-circuit cases. Keeping
# them constant makes evaluation reproducible and matches the tone observed
# in the labeled samples.
ESCALATE_RESPONSE = "Escalate to a human"
OUT_OF_SCOPE_RESPONSE = "I am sorry, this is out of scope from my capabilities"
PLEASANTRY_RESPONSE = "Happy to help"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def require_api_key() -> str:
    """Raise a helpful error if GROQ_API_KEY is missing.

    We don't fail at import time so unit tests can stub the LLM out.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy code/.env.example to code/.env and "
            "fill in your key from https://console.groq.com"
        )
    return GROQ_API_KEY
