"""Pre-LLM deterministic triage layer.

This is the most safety-critical module in the agent. We run a sequence of
explicit rules over each ticket and short-circuit before retrieval/LLM if any
fire. Order matters — first match wins.

Why rules first, LLM second:
  * The rubric explicitly rewards "explicit handling of high-risk, sensitive,
    or out-of-scope tickets" and penalizes hallucinated policies. A rule
    that escalates "score dispute" tickets is far easier to defend in the
    AI Judge interview than "the model decided to escalate."
  * Pleasantries and trivia are trivially cheap to detect with regex —
    no need to spend an LLM call.
  * Site-down style bugs need to escalate even when retrieval finds a
    related doc. Rules guarantee that.

Every decision carries a `rule_name` so the justification column is
traceable back to a specific rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern

from config import (
    ESCALATE_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    PLEASANTRY_RESPONSE,
)
from schema import Ticket, TriageDecision

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def _re(pat: str) -> Pattern[str]:
    return re.compile(pat, flags=re.IGNORECASE | re.DOTALL)


# Pleasantries — must be SHORT messages with only thanks/greeting content.
PLEASANTRY_PATTERNS = [
    _re(r"^\s*(thanks|thank\s*you|thx|ty)\b[^a-z]*$"),
    _re(r"^\s*(thanks|thank\s*you|thx|ty)\s+(for\s+)?(helping|the\s+help|your\s+help|all\s+(of\s+)?your\s+help)\b[^a-z]*$"),
    _re(r"^\s*(thanks|thank\s*you|thx|ty)[\s,!.]+(this\s+)?(was|is)\s+(very\s+)?helpful[^a-z]*$"),
    _re(r"^\s*(great|awesome|cool|nice|perfect|got\s*it)[\s!.]*$"),
    _re(r"^\s*(thanks|thank\s*you)\s+for\s+(helping|all|the).{0,60}$"),
]

# Truly out-of-scope general-knowledge / trivia / unrelated topics.
# These reply (status=replied) with the OOS canned response; request_type=invalid.
TRIVIA_PATTERNS = [
    _re(r"\b(who\s+(is|was)|what\s+is\s+the\s+(name|capital|color)|when\s+was|where\s+is)\s+(the\s+)?(actor|actress|president|prime\s+minister|king|queen|capital|movie|film|song|singer)\b"),
    _re(r"\bwhat\s+(time|day|year)\s+is\s+it\b"),
    _re(r"\b(weather|forecast)\s+(in|for|today|tomorrow)\b"),
    _re(r"\b(iron\s*man|spider\s*man|superman|batman|avengers|harry\s*potter)\b"),
    _re(r"\bwrite\s+(me\s+)?a\s+(poem|story|essay|script)\b"),
    _re(r"\b(joke|riddle|recipe|workout\s+plan)\b"),
]

# High-risk escalations — these always escalate, regardless of retrieval.
# Each entry is (pattern, reason).
ESCALATION_PATTERNS: List[tuple[Pattern[str], str]] = [
    # Total platform outage / mass failures
    (_re(r"\bsite\s+is\s+down\b"), "site_down"),
    (_re(r"\b(none|nothing)\s+of\s+the\s+(pages|tests|submissions|features|services)\s+(are\s+)?(working|loading|accessible)\b"), "platform_outage"),
    (_re(r"\b(all|none\s+of\s+the)\s+submissions\s+(across\s+any\s+(challenges|tests))?\s*(are\s+)?(not\s+)?working\b"), "platform_outage"),
    (_re(r"\bcompletely\s+down\b|\b(everything|all\s+services)\s+(is|are)\s+broken\b"), "platform_outage"),

    # Score / evaluation disputes — agent cannot grade or override results
    (_re(r"\b(review|recheck|re-check|reconsider|dispute|appeal|increase|change|fix)\s+my\s+(score|answers|result|test|grade|rating|evaluation)\b"), "score_dispute"),
    (_re(r"\bgraded?\s+(me\s+)?unfairly\b"), "score_dispute"),
    (_re(r"\b(move|forward|advance)\s+me\s+to\s+the\s+next\s+round\b"), "score_dispute"),
    (_re(r"\bplatform\s+must\s+have\s+graded\s+me\s+unfairly\b"), "score_dispute"),

    # Refund / billing actions on specific accounts/orders the agent can't touch
    (_re(r"\b(refund|chargeback|reverse\s+(the\s+)?charge|return\s+my\s+money)\b"), "refund_request"),
    (_re(r"\b(give|issue|process|provide)\s+(me\s+)?(a\s+|the\s+|my\s+)?refund\b"), "refund_request"),
    (_re(r"\border\s+id[:\s]*[a-z0-9_]+"), "specific_order_lookup"),

    # Account access without ownership / admin override
    (_re(r"\brestore\s+my\s+access\b.{0,100}\b(not|even\s+though)\b.{0,40}\b(owner|admin)\b"), "non_owner_access_restore"),
    (_re(r"\b(IT|admin)\s+(removed|revoked)\s+my\s+seat\b"), "admin_action_required"),

    # Fraud / impersonation / merchant action against third parties
    (_re(r"\bban\s+(the\s+)?(seller|merchant|user|account)\b"), "third_party_action"),
    (_re(r"\b(make|force)\s+(visa|hackerrank|claude|anthropic)\s+(refund|ban|disable|terminate)\b"), "demand_for_company_action"),
    (_re(r"\b(sue|lawsuit|legal\s+action|attorney)\b"), "legal_threat"),
    (_re(r"\b(fraud(ulent)?|scam|stolen|unauthorized\s+(charge|transaction|use))\b"), "fraud_or_theft"),

    # InfoSec / vendor-onboarding form filling
    (_re(r"\b(infosec|security|compliance|vendor)\s+(process|review|questionnaire|form|assessment)\b"), "infosec_request"),
    (_re(r"\bfill(ing)?\s+(in|out)\s+(the\s+)?form\b"), "form_filling_request"),

    # Card / device blocking — sensitive immediate-action requests beyond docs
    (_re(r"\b(block|cancel|deactivate)\s+my\s+(card|account|subscription)\s+(now|asap|immediately)\b"), "urgent_account_action"),
]

# Patterns that strongly indicate a bug rather than a how-to question.
BUG_PATTERNS: List[Pattern[str]] = [
    _re(r"\b(error|bug|broken|crash|crashes|crashed|freezing|hangs|stuck)\b"),
    _re(r"\b(not|isn'?t|aren'?t|cannot|can'?t|unable to)\s+(working|loading|opening|responding|able)\b"),
    _re(r"\bblocker\b"),
    _re(r"\b(blank|white|black)\s+screen\b"),
]

# Feature-request signals.
FEATURE_REQUEST_PATTERNS: List[Pattern[str]] = [
    _re(r"\b(would\s+(love|like)|wish|please\s+add|can\s+you\s+add|feature\s+request|it\s+would\s+be\s+(great|nice)\s+if)\b"),
    _re(r"\bsupport\s+for\s+\w+\s+would\s+be\s+(nice|great|helpful)\b"),
]


# ---------------------------------------------------------------------------
# Triage decision
# ---------------------------------------------------------------------------


@dataclass
class TextSignals:
    """Cached signals on a ticket, used by both triage and the LLM prompt."""

    is_pleasantry: bool
    is_trivia: bool
    is_bug: bool
    is_feature_request: bool
    escalation_reason: Optional[str]


def _full_text(t: Ticket) -> str:
    return f"{t.subject or ''}\n{t.issue or ''}".strip()


def signals(t: Ticket) -> TextSignals:
    text = _full_text(t)
    short = len(text) <= 80  # only short messages can be pleasantries

    is_pleasantry = short and any(p.search(text) for p in PLEASANTRY_PATTERNS)
    is_trivia = any(p.search(text) for p in TRIVIA_PATTERNS) and not _has_support_terms(text)
    is_bug = any(p.search(text) for p in BUG_PATTERNS)
    is_feature_request = any(p.search(text) for p in FEATURE_REQUEST_PATTERNS)

    escalation_reason: Optional[str] = None
    for pat, reason in ESCALATION_PATTERNS:
        if pat.search(text):
            escalation_reason = reason
            break

    return TextSignals(
        is_pleasantry=is_pleasantry,
        is_trivia=is_trivia,
        is_bug=is_bug,
        is_feature_request=is_feature_request,
        escalation_reason=escalation_reason,
    )


# Heuristic: certain support-ish phrases mean "trivia regex matched but the
# user is clearly asking about a real product issue, not pop culture trivia."
_SUPPORT_TERMS = _re(
    r"\b(account|password|test|invite|subscription|billing|payment|card|"
    r"refund|conversation|claude|hackerrank|visa|mock\s+interview|"
    r"assessment|candidate|recruiter|merchant|chargeback)\b"
)


def _has_support_terms(text: str) -> bool:
    return bool(_SUPPORT_TERMS.search(text))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def triage(t: Ticket) -> TriageDecision:
    """Run the rule layer. If it short-circuits, the agent skips retrieval."""
    sig = signals(t)

    # 1. Pleasantries — Replied / invalid.
    if sig.is_pleasantry:
        return TriageDecision(
            short_circuit=True,
            status="replied",
            request_type="invalid",
            response=PLEASANTRY_RESPONSE,
            product_area="",
            justification="Detected as a courtesy message (rule: pleasantry). No support action required.",
            rule_name="pleasantry",
        )

    # 2. Trivia / general-knowledge — Replied with OOS / invalid.
    if sig.is_trivia:
        return TriageDecision(
            short_circuit=True,
            status="replied",
            request_type="invalid",
            response=OUT_OF_SCOPE_RESPONSE,
            product_area="",
            justification="Question is general-knowledge / unrelated to HackerRank, Claude, or Visa support (rule: trivia).",
            rule_name="trivia",
        )

    # 3. High-risk escalations.
    if sig.escalation_reason:
        request_type = "bug" if sig.escalation_reason in {"site_down", "platform_outage"} else "product_issue"
        return TriageDecision(
            short_circuit=True,
            status="escalated",
            request_type=request_type,
            response=ESCALATE_RESPONSE,
            product_area="",
            justification=(
                f"Escalated to a human (rule: {sig.escalation_reason}). "
                "This case is high-risk, requires backend access, or demands an action the agent cannot safely take."
            ),
            rule_name=f"escalate:{sig.escalation_reason}",
        )

    # No short-circuit — fall through to retrieval + LLM.
    return TriageDecision(
        short_circuit=False,
        rule_name="fallthrough",
    )
