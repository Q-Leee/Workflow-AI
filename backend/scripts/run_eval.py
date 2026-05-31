#!/usr/bin/env python3
"""Run a small retrieval/answer eval set against the API or in-process RAG.

Example eval file (eval/questions.json):
[
  {"document_id": "...", "question": "What is the leave policy?", "expect_contains": ["annual", "leave"]}
]

Usage (from backend/):
  python scripts/run_eval.py --eval eval/questions.json --token YOUR_JWT
  python scripts/run_eval.py --eval eval/questions.json --in-process --user-id USER_UUID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as: python scripts/run_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.config import settings
from app.services import rag


def _hit(expect: list[str], text: str) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in expect)


def run_in_process(cases: list[dict], user_id: str) -> None:
    ok_retrieval = 0
    ok_answer = 0
    for i, case in enumerate(cases, 1):
        q = case["question"]
        doc_id = case.get("document_id")
        expect = case.get("expect_contains") or []
        sources = rag.retrieve_and_rerank(
            user_id=user_id,
            question=q,
            document_id=doc_id,
            filename=None,
            doc_type=None,
            page_min=None,
            page_max=None,
            top_k=case.get("top_k", settings.query_top_k_default),
            do_rerank=True,
        )
        blob = " ".join(s.text for s in sources)
        rel = rag.best_source_relevance(sources)
        retrieved = _hit(expect, blob) if expect else rel >= settings.answer_min_relevance
        answer = rag.generate_answer_ollama(q, sources) if sources else None
        answered = _hit(expect, answer or "") if expect and answer else bool(
            answer and answer != rag.LOW_CONFIDENCE_ANSWER
        )
        ok_retrieval += int(retrieved)
        ok_answer += int(answered)
        print(
            f"[{i}] retrieval={'OK' if retrieved else 'MISS'} "
            f"answer={'OK' if answered else 'MISS'} rel={rel:.2f} q={q[:60]!r}"
        )
    n = len(cases)
    print(f"\nRetrieval: {ok_retrieval}/{n} ({100 * ok_retrieval / max(n, 1):.0f}%)")
    print(f"Answer:    {ok_answer}/{n} ({100 * ok_answer / max(n, 1):.0f}%)")


def run_api(cases: list[dict], base_url: str, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    ok = 0
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        for i, case in enumerate(cases, 1):
            r = client.post(
                "/query",
                json={
                    "question": case["question"],
                    "document_id": case.get("document_id"),
                    "top_k": case.get("top_k", settings.query_top_k_default),
                    "use_llm": True,
                },
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            expect = case.get("expect_contains") or []
            blob = " ".join(s.get("text", "") for s in data.get("sources") or [])
            blob += " " + (data.get("answer") or "")
            hit = _hit(expect, blob) if expect else bool(data.get("answer"))
            ok += int(hit)
            print(f"[{i}] {'OK' if hit else 'MISS'} q={case['question'][:60]!r}")
    n = len(cases)
    print(f"\nCombined hit: {ok}/{n} ({100 * ok / max(n, 1):.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="WorkFlow AI eval runner")
    p.add_argument("--eval", required=True, help="Path to JSON eval cases")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--token", help="JWT for API mode")
    p.add_argument("--in-process", action="store_true")
    p.add_argument("--user-id", help="User UUID for in-process mode")
    args = p.parse_args()

    cases = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise SystemExit("Eval file must be a JSON array")

    if args.in_process:
        if not args.user_id:
            raise SystemExit("--user-id required for --in-process")
        run_in_process(cases, args.user_id)
    else:
        if not args.token:
            raise SystemExit("--token required for API mode (or use --in-process)")
        run_api(cases, args.base_url.rstrip("/"), args.token)


if __name__ == "__main__":
    main()
