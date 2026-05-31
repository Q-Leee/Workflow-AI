import json
import logging

from app.db.database import get_conn

logger = logging.getLogger("workflow.audit")


def log_action(*, user_id: str | None, action: str, detail: dict | str | None = None) -> None:
    detail_str = detail if isinstance(detail, str) else json.dumps(detail or {}, ensure_ascii=False)
    logger.info("action=%s user_id=%s detail=%s", action, user_id, detail_str)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO audit_logs (user_id, action, detail) VALUES (?, ?, ?)",
                (user_id, action, detail_str),
            )
    except Exception:
        logger.exception("Failed to persist audit log")
