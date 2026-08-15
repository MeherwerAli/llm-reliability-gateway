import re
from dataclasses import dataclass
from typing import Literal

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
SECRET = re.compile(r"(?i)\b(?:api[_ -]?key|secret|token)\b\s*[:=]\s*[A-Za-z0-9._-]{12,}")
INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?developer\s+mode", re.IGNORECASE),
)


@dataclass(frozen=True)
class PolicyDecision:
    action: Literal["allow", "flag", "block"]
    reasons: tuple[str, ...]
    redacted_text: str


def redact_pii(text: str) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    redacted = text
    for name, pattern, replacement in (
        ("email", EMAIL, "[REDACTED_EMAIL]"),
        ("phone", PHONE, "[REDACTED_PHONE]"),
        ("payment_card", CARD, "[REDACTED_CARD]"),
        ("secret", SECRET, "[REDACTED_SECRET]"),
    ):
        if pattern.search(redacted):
            reasons.append(name)
            redacted = pattern.sub(replacement, redacted)
    return redacted, tuple(reasons)


def assess_input(text: str) -> PolicyDecision:
    redacted, reasons = redact_pii(text)
    injection = any(pattern.search(text) for pattern in INJECTION_PATTERNS)
    all_reasons = list(reasons)
    if injection:
        all_reasons.append("prompt_injection_indicator")
    action: Literal["allow", "flag", "block"] = "flag" if all_reasons else "allow"
    return PolicyDecision(action=action, reasons=tuple(all_reasons), redacted_text=redacted)


def assess_output(text: str) -> PolicyDecision:
    redacted, reasons = redact_pii(text)
    leak = SECRET.search(text) is not None
    all_reasons = list(reasons)
    if leak and "secret" not in all_reasons:
        all_reasons.append("secret")
    action: Literal["allow", "flag", "block"] = (
        "block" if leak else ("flag" if all_reasons else "allow")
    )
    return PolicyDecision(action=action, reasons=tuple(all_reasons), redacted_text=redacted)
