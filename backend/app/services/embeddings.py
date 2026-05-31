import httpx

from app.config import settings

_bi_encoder = None


def _get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer

        _bi_encoder = SentenceTransformer(settings.embedding_model_local)
    return _bi_encoder


def embed_texts(texts: list[str]) -> list[list[float]]:
    if settings.embedding_provider == "ollama":
        return _embed_ollama(texts)
    enc = _get_bi_encoder()
    vectors = enc.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.astype(float).tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embeddings"
    with httpx.Client(timeout=120.0) as client:
        for t in texts:
            r = client.post(
                url,
                json={"model": settings.ollama_embed_model, "prompt": t},
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Unexpected Ollama embeddings response")
            out.append(emb)
    return out
