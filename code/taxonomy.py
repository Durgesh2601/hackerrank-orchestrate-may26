"""Product-area taxonomy.

Maps corpus folder paths into the snake_case product_area labels we emit to
output.csv. The mapping rules below are derived from the labeled
sample_support_tickets.csv (e.g. "hackerrank_community" -> "community",
"privacy-and-legal" -> "privacy", visa/support/consumer/travel-support ->
"travel_support") plus a uniform kebab→snake fallback for everything else.

Centralizing this here keeps corpus loading dumb and the taxonomy auditable.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Dict, Iterable, List

# Hand-coded overrides observed in the labeled samples or chosen for clarity.
# Key = (company, folder_name); value = product_area label.
_OVERRIDES: Dict[tuple[str, str], str] = {
    ("hackerrank", "hackerrank_community"): "community",
    ("hackerrank", "general-help"): "general_help",
    ("claude", "privacy-and-legal"): "privacy",
    ("claude", "claude-api-and-console"): "api_and_console",
    ("claude", "claude-code"): "claude_code",
    ("claude", "claude-desktop"): "claude_desktop",
    ("claude", "claude-for-education"): "education",
    ("claude", "claude-for-government"): "government",
    ("claude", "claude-for-nonprofits"): "nonprofits",
    ("claude", "claude-in-chrome"): "claude_in_chrome",
    ("claude", "claude-mobile-apps"): "mobile_apps",
    ("claude", "amazon-bedrock"): "amazon_bedrock",
    ("claude", "identity-management-sso-jit-scim"): "identity_management",
    ("claude", "pro-and-max-plans"): "plans",
    ("claude", "team-and-enterprise-plans"): "team_and_enterprise",
}


def _kebab_to_snake(s: str) -> str:
    return s.replace("-", "_")


def product_area_for(company: str, rel_path: str) -> str:
    """Return the canonical product_area string for a corpus file.

    rel_path is relative to data/, e.g. "hackerrank/screen/test-settings/foo.md".
    """
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return "general_support"
    # parts[0] is the company. We want the next meaningful folder.
    if len(parts) == 1:
        # File directly at data/{company} (e.g. visa/index.md). Fallback.
        return "general_support"

    # Visa nests under support/{consumer,merchant,small-business}/...
    if company == "visa":
        # parts: ("visa", "support", "<segment>", ...)
        if len(parts) >= 4 and parts[1] == "support":
            seg2 = parts[2]
            seg3 = parts[3]
            # If the next layer is a sub-folder (not a file), prefer it as
            # it's more specific (e.g. consumer/travel-support/...).
            if not seg3.endswith(".md"):
                return _kebab_to_snake(seg3)
            return _kebab_to_snake(seg2) if seg2 != "consumer" else "general_support"
        if len(parts) >= 3 and parts[1] == "support":
            # visa/support/consumer.md, visa/support/merchant.md, etc.
            stem = parts[2].removesuffix(".md")
            return _kebab_to_snake(stem) if stem != "consumer" else "general_support"
        return "general_support"

    folder = parts[1]
    key = (company, folder)
    if key in _OVERRIDES:
        return _OVERRIDES[key]
    return _kebab_to_snake(folder)


def vocabulary_for(company: str, all_chunk_paths: Iterable[str]) -> List[str]:
    """The set of product_area labels that exist in the corpus for a company."""
    seen = set()
    for p in all_chunk_paths:
        if p.split("/", 1)[0] != company:
            continue
        seen.add(product_area_for(company, p))
    return sorted(seen)


def all_vocabulary(all_chunk_paths: Iterable[str]) -> Dict[str, List[str]]:
    """All product_area labels grouped by company."""
    paths = list(all_chunk_paths)
    return {
        c: vocabulary_for(c, paths) for c in ("hackerrank", "claude", "visa")
    }
