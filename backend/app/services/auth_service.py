import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
from jose import JWTError, jwt

from app.config import settings
from app.db.database import get_conn


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False

def create_user(email: str, password: str) -> dict:
    user_id = str(uuid.uuid4())
    email_norm = email.strip().lower()
    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email_norm,)).fetchone()
        if existing:
            raise ValueError("Email already registered")
        conn.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
            (user_id, email_norm, hash_password(password)),
        )
    return {"id": user_id, "email": email_norm}


def authenticate_user(email: str, password: str) -> dict | None:
    email_norm = email.strip().lower()
    with get_conn() as conn:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email_norm,)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "email": row["email"]}


def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


def get_user_by_id(user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "email": row["email"]}


def register_document(
    *,
    user_id: str,
    document_id: str,
    filename: str,
    doc_type: str,
    chunks: int,
    content_hash: str | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, user_id, filename, doc_type, chunks_indexed, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, user_id, filename, doc_type, chunks, content_hash),
        )


def find_document_by_hash(*, user_id: str, doc_type: str, content_hash: str) -> str | None:
    if not content_hash:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE user_id = ? AND doc_type = ? AND content_hash = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id, doc_type, content_hash),
        ).fetchone()
    return row["id"] if row else None


def user_owns_document(user_id: str, document_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        ).fetchone()
    return row is not None


def list_documents(user_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, doc_type, chunks_indexed, created_at FROM documents "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(*, user_id: str, document_id: str) -> bool:
    if not user_owns_document(user_id, document_id):
        return False
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
        conn.execute(
            "DELETE FROM query_history WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        )
    from app.services import vector_store

    vector_store.delete_document_chunks(document_id)

    upload_dir = Path(settings.upload_dir)
    if upload_dir.exists():
        for p in upload_dir.glob(f"{document_id}_*"):
            try:
                p.unlink()
            except OSError:
                pass
    return True


def save_query_history(
    *,
    user_id: str,
    document_id: str | None,
    question: str,
    answer: str | None,
    source_count: int,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO query_history (user_id, document_id, question, answer, source_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, document_id, question, answer, source_count),
        )
        return int(cur.lastrowid)


def list_query_history(
    *,
    user_id: str,
    document_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    with get_conn() as conn:
        if document_id:
            rows = conn.execute(
                "SELECT id, document_id, question, answer, source_count, created_at "
                "FROM query_history WHERE user_id = ? AND document_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, document_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, document_id, question, answer, source_count, created_at "
                "FROM query_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]
