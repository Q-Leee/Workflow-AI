"""
Sentence-level evidence extraction from resume chunks for requirement matching.
"""
from __future__ import annotations

import re

from app.services import skill_ontology

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n(?=[•●▪\-\*]\s)|\n{2,}")
_KEYWORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}")
_STOP = frozenset(
    {
        "with",
        "and",
        "the",
        "for",
        "experience",
        "strong",
        "including",
        "related",
        "ability",
        "understanding",
    }
)


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    raw = text.replace("\r\n", "\n")
    parts: list[str] = []
    for block in _SENTENCE_SPLIT.split(raw):
        block = re.sub(r"^[•●▪\-\*]\s*", "", block.strip())
        if len(block) >= 20:
            parts.append(block)
    if not parts and raw.strip():
        parts.append(raw.strip())
    return parts


def _token_overlap(req_text: str, sentence: str) -> float:
    req_tokens = {
        t.lower()
        for t in _KEYWORD_RE.findall(req_text)
        if len(t) > 3 and t.lower() not in _STOP
    }
    if not req_tokens:
        return 0.0
    sent_lower = sentence.lower()
    hit = sum(1 for t in req_tokens if t in sent_lower)
    return hit / len(req_tokens)


def score_sentence(sentence: str, req_text: str) -> float:
    ont = skill_ontology.ontology_overlap_score(req_text, sentence)
    tok = _token_overlap(req_text, sentence)
    return min(1.0, 0.55 * ont + 0.45 * tok)


def extract_best_evidence(
    chunk_text: str,
    req_text: str,
    *,
    max_chars: int = 420,
) -> str | None:
    """Pick the best 1–2 sentences that justify the match."""
    sentences = split_sentences(chunk_text)
    if not sentences:
        return None
    scored = [(score_sentence(s, req_text), s) for s in sentences]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_sent = scored[0]
    if best_score < 0.12:
        return chunk_text[:max_chars].strip() if chunk_text else None

    parts = [best_sent]
    total = len(best_sent)
    for score, sent in scored[1:3]:
        if score < 0.20 or total + len(sent) > max_chars:
            break
        if sent != best_sent:
            parts.append(sent)
            total += len(sent) + 1
    return " ".join(parts)[:max_chars].strip()
