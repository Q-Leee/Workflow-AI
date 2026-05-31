"""
Reset a WorkFlow AI user password (local dev — no email reset in app yet).

Usage (from backend folder):
  .venv\\Scripts\\python scripts\\reset_password.py --email you@example.com --password NewPass123
  .venv\\Scripts\\python scripts\\reset_password.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.database import get_conn, init_db
from app.services.auth_service import hash_password


def list_users() -> None:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT email, created_at FROM users ORDER BY created_at").fetchall()
    if not rows:
        print("No users in database:", settings.sqlite_path)
        return
    print(f"Users in {settings.sqlite_path}:")
    for r in rows:
        print(f"  - {r['email']}  (created {r['created_at']})")


def reset_password(email: str, password: str) -> None:
    if len(password) < 6:
        print("Password must be at least 6 characters.")
        sys.exit(1)
    init_db()
    email_norm = email.strip().lower()
    hashed = hash_password(password)
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (hashed, email_norm),
        )
        if cur.rowcount == 0:
            print(f"No user with email: {email_norm}")
            print("Run with --list to see registered emails.")
            sys.exit(1)
    print(f"Password updated for {email_norm}")
    print("You can log in at the app with the new password.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset WorkFlow AI local user password")
    parser.add_argument("--email", help="Account email")
    parser.add_argument("--password", help="New password (min 6 chars)")
    parser.add_argument("--list", action="store_true", help="List all users")
    args = parser.parse_args()
    if args.list:
        list_users()
        return
    if not args.email or not args.password:
        parser.print_help()
        sys.exit(1)
    reset_password(args.email, args.password)


if __name__ == "__main__":
    main()
