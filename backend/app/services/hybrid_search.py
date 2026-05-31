"""Reciprocal Rank Fusion of vector search and BM25 keyword search."""

import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import settings

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#./-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _rrf_merge(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rlist in ranked_lists:
        for rank, cid in enumerate(rlist):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_rank(
    question: str,
    *,
    chunk_ids: list[str],
    documents: list[str],
    vector_scores: list[float],
    top_k: int,
) -> list[int]:
    """Return indices into chunk_ids/documents ordered by fused relevance."""
    if not chunk_ids:
        return []

    n = len(chunk_ids)
    if not settings.hybrid_search_enabled or n < 2:
        vec_order = sorted(range(n), key=lambda i: vector_scores[i], reverse=True)
        return vec_order[:top_k]

    vec_ranked = [chunk_ids[i] for i in sorted(range(n), key=lambda i: vector_scores[i], reverse=True)]

    tokenized_corpus = [_tokenize(d) for d in documents]
    if not any(tokenized_corpus):
        return list(range(min(top_k, n)))

    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = _tokenize(question)
    if not q_tokens:
        fused = _rrf_merge([vec_ranked])
    else:
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranked = [chunk_ids[i] for i in bm25_order]
        fused = _rrf_merge([vec_ranked, bm25_ranked])

    id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
    out: list[int] = []
    for cid, _ in fused:
        if cid in id_to_idx:
            out.append(id_to_idx[cid])
        if len(out) >= top_k:
            break
    return out
