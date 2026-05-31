"""
JD requirement normalization: structured dimensions + search queries for retrieval.
"""
from __future__ import annotations

import re

from app.services import skill_ontology

_CATEGORY_TO_DIMENSION = {
    "skill": "skill",
    "education": "domain",
    "experience": "outcome",
    "other": "domain",
}

_DOMAIN_SIGNAL = re.compile(
    r"\b(banking|financial|healthcare|retail|cinema|media|logistics|"
    r"enterprise|saas|fintech|insurance)\b",
    re.I,
)
_TOOL_SIGNAL = re.compile(
    r"\b(gcp|aws|azure|kubernetes|docker|terraform|snowflake|dbt|"
    r"playwright|chromadb?|postgres|mysql|react|java|python)\b",
    re.I,
)


def infer_dimension(req: dict) -> str:
    text = str(req.get("text") or "")
    cat = str(req.get("category") or "other").lower()
    if _TOOL_SIGNAL.search(text):
        return "tool"
    if _DOMAIN_SIGNAL.search(text):
        return "domain"
    if re.search(r"\b(deliver|ship|production|years? of|deployed|led)\b", text, re.I):
        return "outcome"
    return _CATEGORY_TO_DIMENSION.get(cat, "skill")


def enrich_requirement(req: dict) -> dict:
    """Add dimension + search_queries for retrieval pipeline."""
    text = str(req.get("text") or "").strip()
    category = str(req.get("category") or "other")
    priority = str(req.get("priority") or "required")
    dimension = str(req.get("dimension") or infer_dimension(req))
    existing_queries = req.get("search_queries")
    if isinstance(existing_queries, list) and existing_queries:
        queries = [str(q) for q in existing_queries if q]
    else:
        queries = skill_ontology.expand_search_queries(text, category=category)
    return {
        "text": text,
        "category": category,
        "priority": priority,
        "dimension": dimension,
        "search_queries": queries,
    }


def enrich_requirements(requirements: list[dict]) -> list[dict]:
    return [enrich_requirement(r) for r in requirements if r.get("text")]
