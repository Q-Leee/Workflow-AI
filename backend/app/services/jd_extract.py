"""
Generic JD requirement extraction — layout-agnostic section parsing.

Avoids per-recruiter/per-company regex patches; uses broad section headers,
technical-line detection, and prose splitting.
"""
from __future__ import annotations

import re

from app.services import requirement_knowledge

# --- Section headers (any layout) ---
_SCORABLE_SECTION = re.compile(
    r"^(?:"
    r"(?:job\s+)?(?:key\s+)?(?:skills?(?:\s*[&/]?\s*(?:and|&)\s*experience)?|requirements?|qualifications?)|"
    r"must have|mandatory|essential(?:\s+criteria)?|minimum\s+qualifications?|"
    r"what you.?ll (?:bring|need)|what we.?re looking for|"
    r"nice to have|preferred(?:\s+qualifications?)?|desired|bonus(?:\s+points?)?"
    r")\b",
    re.IGNORECASE,
)

_PREFERRED_SECTION = re.compile(
    r"\b(nice to have|preferred|bonus|desirable|optional)\b",
    re.IGNORECASE,
)

_STOP_SECTION = re.compile(
    r"^(?:"
    r"(?:job\s+)?(?:benefits?|compensation|salary|perks?|what.?s on offer|how to apply|"
    r"equal opportunity|about (?:the )?(?:job|role|company|us)|company profile|"
    r"role overview|project workflow|create an account|payments? are issued)"
    r")\b",
    re.IGNORECASE,
)

_DUTY_SECTION = re.compile(
    r"^(?:"
    r"(?:job\s+)?(?:key\s+)?responsibilities|what you.?ll be doing|duties|about the job"
    r")\b",
    re.IGNORECASE,
)

# --- Line classification ---
_TECHNICAL_LINE = re.compile(
    r"\b("
    r"software|engineer|developer|backend|frontend|full[- ]?stack|database|sql|nosql|"
    r"python|javascript|typescript|java|go\b|golang|react|node|api|rest|cloud|"
    r"llm|ai\b|agent|rag|embed|vector|machine learning|automation|integrat|"
    r"architect|modular|production|deploy|degree|bachelor|master|graduate|"
    r"experience|years?|git|docker|kubernetes|eks|helm|terraform|ansible|"
    r"cloudwatch|next\.?js|laravel|expo|php|forge|micro[- ]?services?|aws|azure|gcp|"
    r"supabase|devops|gitops|ci/?cd|"
    r"evaluat|rubric|prompt|security|privacy|session|firmware|embedded|c programming|"
    r"typescript|php|symfony|testing|ci/?cd|"
    r"data structures|algorithms|computer science|c\+\+|c#|programming|object[- ]oriented|oop|design patterns|system design|software development|science|engineering"
    r")\b",
    re.IGNORECASE,
)

_SOFT_FIT_ONLY = re.compile(
    r"^(product minded|enthusiastic about fully remote|self[- ]directed|speaking up|"
    r"you.?re curious|who you are|cultural fit|team player)\b",
    re.IGNORECASE,
)

_BOILERPLATE = re.compile(
    r"(equal opportunity|selection decisions are based solely|https?://|www\.|"
    r"linkedin\.com|salary range:|',?\s*000\s*[-–—]\s*\$|payments are issued weekly)",
    re.IGNORECASE,
)

_PREFERRED_IN_LINE = re.compile(
    r"\b(bonus|nice to have|preferred|desirable|optional|not essential)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+|\s+[-–—]\s+")


def _clean_line(line: str) -> str:
    t = line.strip()
    t = re.sub(r"^[-•*·]\s*", "", t)
    t = re.sub(r"^\d+[\.\)]\s*", "", t).strip()
    return t


def _infer_priority(text: str, *, section_preferred: bool) -> str:
    if section_preferred or _PREFERRED_IN_LINE.search(text):
        return "preferred"
    return "required"


def _infer_category(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(bachelor|master|degree|graduate|diploma|computer science|related)\b", lower):
        return "education"
    if re.search(r"\b(\d+\+?\s*years?|production|live environment|non-mocked)\b", lower):
        return "experience"
    if _TECHNICAL_LINE.search(text):
        return "skill"
    return "other"


def _is_scorable_line(text: str) -> bool:
    t = text.strip()
    if len(t) < 15 or len(t) > 420:
        return False
    if requirement_knowledge.is_junk_requirement(t):
        return False
    if _BOILERPLATE.search(t) or _SOFT_FIT_ONLY.search(t):
        return False
    if re.search(r"^(create an account|complete the onboarding|start earning)\b", t, re.I):
        return False
    return bool(_TECHNICAL_LINE.search(t))


def _split_prose(text: str) -> list[str]:
    """Split long qualification paragraphs into scorable clauses."""
    t = text.strip()
    if len(t) <= 100:
        return [t] if _is_scorable_line(t) else []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(t) if p.strip()]
    out: list[str] = []
    for p in parts:
        p = re.sub(r"^(and|or)\s+", "", p, flags=re.I).strip()
        if _is_scorable_line(p):
            out.append(p)
    return out if out else ([t] if _is_scorable_line(t) else [])


def _technical_responsibility(text: str) -> bool:
    """Include responsibility bullets that describe technical work (eval, architecture, AI)."""
    if not _TECHNICAL_LINE.search(text):
        return False
    return bool(
        re.search(
            r"\b(llm|ai agent|agent|evaluat|rubric|architect|modular|prompt|inject|"
            r"debug|trace|integration|backend|software|code|system interaction|"
            r"multi-turn|production|automation)\b",
            text,
            re.I,
        )
    )


def extract_requirements(text: str, *, max_items: int = 18) -> list[dict]:
    """
    Generic extractor: scorable sections → bullets/prose; technical duties optional.
    """
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").splitlines() if ln.strip()]
    bullets: list[dict] = []
    mode: str | None = None  # scorable | duty | stop
    section_preferred = False

    for ln in lines:
        if _STOP_SECTION.match(ln):
            mode = "stop"
            section_preferred = False
            continue
        if _SCORABLE_SECTION.match(ln):
            mode = "scorable"
            section_preferred = bool(_PREFERRED_SECTION.search(ln))
            continue
        if _DUTY_SECTION.match(ln):
            mode = "duty"
            section_preferred = False
            continue

        if mode == "stop":
            continue

        cleaned = _clean_line(ln)
        if not cleaned or len(cleaned) < 12:
            continue

        if mode == "duty":
            if _is_scorable_line(cleaned):
                bullets.append(
                    {
                        "text": cleaned,
                        "priority": _infer_priority(cleaned, section_preferred=False),
                    }
                )
            continue

        # Prose outside named sections (e.g. "The Role" intro) — still scorable if technical.
        if mode is None and _is_scorable_line(cleaned):
            bullets.append(
                {
                    "text": cleaned,
                    "priority": _infer_priority(cleaned, section_preferred=False),
                }
            )
            continue

        if mode != "scorable":
            continue

        pieces = _split_prose(cleaned) if len(cleaned) > 60 else [cleaned]
        for piece in pieces:
            if not _is_scorable_line(piece):
                continue
            bullets.append(
                {
                    "text": piece,
                    "priority": _infer_priority(piece, section_preferred=section_preferred),
                }
            )

    # Fallback: scan whole doc for obvious qualification sentences if section headers failed
    if len(bullets) < 3:
        blob = " ".join(_clean_line(ln) for ln in lines)
        for m in re.finditer(
            r"([A-Z][^.!?]{20,220}\b(?:experience|degree|languages?|SQL|LLM|AI|software|"
            r"backend|integration|architect|production)[^.!?]*[.!?])",
            blob,
        ):
            piece = m.group(1).strip()
            if _is_scorable_line(piece):
                bullets.append(
                    {
                        "text": piece,
                        "priority": _infer_priority(piece, section_preferred=False),
                    }
                )

    raw = [
        {
            "text": b["text"],
            "category": _infer_category(b["text"]),
            "priority": b["priority"],
        }
        for b in bullets
    ]
    return requirement_knowledge.dedupe_requirements(raw)[:max_items]
