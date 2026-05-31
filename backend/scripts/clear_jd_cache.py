"""Clear cached JD requirement parses (run after parser/scoring updates)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.database import get_conn, init_db


def main() -> None:
    init_db()
    with get_conn() as conn:
        n = conn.execute("DELETE FROM jd_requirement_cache").rowcount
    print(f"Cleared {n} JD parse cache row(s) in {settings.sqlite_path}")


if __name__ == "__main__":
    main()
