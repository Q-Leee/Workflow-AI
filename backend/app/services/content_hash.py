import hashlib
import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(normalize_text(text).encode("utf-8"))
