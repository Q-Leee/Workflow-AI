import uuid
from pathlib import Path

import chromadb
from chromadb.api.types import Where

from app.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client


def collection_full_name() -> str:
    return f"{settings.chroma_collection}_{settings.embedding_provider}"


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=collection_full_name(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    *,
    user_id: str,
    document_id: str,
    filename: str,
    doc_type: str,
    texts: list[str],
    pages: list[int],
    chunk_indices: list[int],
    sections: list[str],
    embeddings: list[list[float]],
) -> list[str]:
    col = get_collection()
    ids = [f"{document_id}:{chunk_indices[i]}" for i in range(len(texts))]
    metadatas = [
        {
            "user_id": user_id,
            "document_id": document_id,
            "filename": filename,
            "doc_type": doc_type,
            "page": int(pages[i]),
            "chunk_index": int(chunk_indices[i]),
            "section": str(sections[i])[:64],
        }
        for i in range(len(texts))
    ]
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return ids


def build_where_filter(
    *,
    user_id: str,
    document_id: str | None,
    filename: str | None,
    doc_type: str | None,
    page_min: int | None,
    page_max: int | None,
    section: str | None = None,
) -> Where:
    clauses: list = [{"user_id": user_id}]
    if document_id:
        clauses.append({"document_id": document_id})
    if filename:
        clauses.append({"filename": filename})
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if page_min is not None:
        clauses.append({"page": {"$gte": page_min}})
    if page_max is not None:
        clauses.append({"page": {"$lte": page_max}})
    if section:
        clauses.append({"section": section})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query_similar(
    query_embedding: list[float],
    *,
    top_k: int,
    where: Where,
) -> tuple[list[str], list[str], list[dict], list[float]]:
    col = get_collection()
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, 100),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0] or []
    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []
    distances = (res.get("distances") or [[]])[0] or []
    scores = [1.0 - float(d) for d in distances] if distances else [0.0] * len(ids)
    return ids, docs, metas, scores


def fetch_chunks(
    where: Where,
    *,
    limit: int = 300,
) -> tuple[list[str], list[str], list[dict]]:
    col = get_collection()
    res = col.get(where=where, include=["documents", "metadatas"], limit=limit)
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return ids, docs, metas


def delete_document_chunks(document_id: str) -> None:
    col = get_collection()
    try:
        col.delete(where={"document_id": document_id})
    except Exception:
        pass


def new_document_id() -> str:
    return str(uuid.uuid4())
