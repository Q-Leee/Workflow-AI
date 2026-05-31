import logging
import math

from app.config import settings

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.rerank_model)
    return _cross_encoder


def relevance_0_1(*, rerank_score: float | None, vector_score: float | None) -> float:
    """Map cross-encoder logits and/or cosine similarity to a 0–1 relevance score."""
    vec = 0.0
    if vector_score is not None:
        vec = max(0.0, min(1.0, float(vector_score)))
    if rerank_score is not None:
        # ms-marco-style cross-encoders emit unbounded logits (often ~ -12 .. +12).
        ce = 1.0 / (1.0 + math.exp(-float(rerank_score)))
        # Informal questions (e.g. "what skill he got") often score low on CE but high on vectors.
        return max(ce, vec)
    return vec


def rerank_pairs(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    ce = _get_cross_encoder()
    pairs = [[query, p] for p in passages]
    scores = ce.predict(pairs, show_progress_bar=False)
    # numpy scalar or float
    return [float(s) for s in scores]
