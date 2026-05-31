"""
meeting_service.py
─────────────────
Accurate meeting transcript → summary + action items.

Pipeline:
  1. preprocess_transcript  – normalise whitespace while keeping speaker turns
  2. chunk_transcript       – split long transcripts into overlapping segments
  3. summarize_chunk        – run LLM on each segment → partial results
  4. merge_summaries        – if multiple chunks, do a second-pass merge call
  5. parse + validate       – robust JSON extraction with one automatic retry
"""

import logging
import re
from pathlib import Path

from app.config import settings
from app.models.schemas import ActionItem, MeetingSummaryResponse
from app.services import llm
from app.services.document_extract import SUPPORTED_EXTENSIONS, extract_text_from_file

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# File extraction
# ─────────────────────────────────────────────

def extract_text_from_upload(path: Path, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return extract_text_from_file(path, filename)
    raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────

def preprocess_transcript(text: str) -> str:
    """
    Clean the raw transcript while preserving speaker-turn structure.
    - Collapse multiple spaces / tabs into one space (per line)
    - Collapse 3+ consecutive blank lines into 2
    - Strip leading/trailing whitespace
    """
    lines = text.splitlines()
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in lines]
    joined = "\n".join(cleaned_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


# ─────────────────────────────────────────────
# Chunking for long transcripts
# ─────────────────────────────────────────────

def chunk_transcript(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Split transcript into chunks of ≤ max_chars characters.
    Tries to break at paragraph boundaries; falls back to hard split.
    Consecutive chunks share `overlap_chars` characters for context continuity.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        segment = text[start:end]

        # Try to break at the last paragraph boundary inside the segment
        if end < len(text):
            para_break = segment.rfind("\n\n")
            if para_break > max_chars // 3:  # must be past 1/3 of window
                end = start + para_break + 2  # include the two newlines
                segment = text[start:end]

        chunks.append(segment.strip())
        next_start = end - overlap_chars
        # Avoid infinite loop
        if next_start <= start:
            next_start = start + max(1, max_chars - overlap_chars)
        start = next_start

    return [c for c in chunks if c]


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

_CHUNK_SYSTEM = (
    "You are an expert meeting analyst. Given a portion of a meeting transcript, extract:\n"
    "1. A concise summary (key topics discussed, decisions made, outcomes). 3-6 sentences.\n"
    "2. All concrete action items mentioned. An action item is something a person is expected to DO.\n"
    "   - Skip vague statements ('let's think about it', 'maybe someday').\n"
    "   - owner: person's name if explicitly stated, else null.\n"
    "   - due: exact date or deadline if mentioned, else null.\n\n"
    "LANGUAGE RULE: Respond STRICTLY in the SAME language as the input transcript.\n"
    "  - Korean transcript → Korean output (한국어로 답변)\n"
    "  - English transcript → English output\n"
    "  - Mixed → use the dominant language\n"
    "  - NEVER output Chinese unless the input is Chinese.\n\n"
    "Output ONLY a valid JSON object with NO markdown fences, NO extra text:\n"
    '{"summary": "<summary text>", '
    '"action_items": [{"task": "<task>", "owner": "<name or null>", "due": "<date or null>"}]}'
)

_MERGE_SYSTEM = (
    "You are an expert meeting analyst. You have received partial summaries and action item lists "
    "from different sections of the same meeting transcript.\n"
    "Your job:\n"
    "1. Write ONE unified, coherent summary that captures the entire meeting. 4-7 sentences.\n"
    "2. Merge ALL action items into a single de-duplicated list.\n"
    "   - Remove exact duplicates or near-duplicates.\n"
    "   - Preserve all unique tasks, owners, and deadlines.\n\n"
    "LANGUAGE RULE: Match the language of the partial summaries (same language as the transcript).\n"
    "  NEVER output Chinese unless the partial summaries are in Chinese.\n\n"
    "Output ONLY a valid JSON object with NO markdown fences, NO extra text:\n"
    '{"summary": "<unified summary>", '
    '"action_items": [{"task": "<task>", "owner": "<name or null>", "due": "<date or null>"}]}'
)


# ─────────────────────────────────────────────
# LLM call helpers
# ─────────────────────────────────────────────

def _call_with_retry(system: str, user_content: str, max_retries: int = 1) -> dict | None:
    """
    Call LLM and parse JSON. On failure, retry once with an explicit correction prompt.
    Uses meeting_chat() which respects MEETING_CHAT_MODEL and MEETING_TEMPERATURE.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    raw = llm.meeting_chat(messages)
    parsed = llm.extract_json_block(raw or "")
    if parsed:
        return parsed

    logger.warning("JSON parse failed on first attempt. Retrying with correction prompt.")
    for attempt in range(max_retries):
        correction_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": raw or "",
            },
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Please output ONLY the raw JSON object with NO markdown, NO extra text. "
                    "Start your response with { and end with }."
                ),
            },
        ]
        raw = llm.meeting_chat(correction_messages, temperature=0.0)
        parsed = llm.extract_json_block(raw or "")
        if parsed:
            logger.info("JSON recovered on retry %d", attempt + 1)
            return parsed

    logger.error("LLM returned non-JSON after %d retries. Raw: %s", max_retries, (raw or "")[:300])
    return None


def _parse_result(parsed: dict, fallback_summary: str) -> tuple[str, list[ActionItem]]:
    """Extract summary and action items from a parsed JSON dict."""
    summary = str(parsed.get("summary") or fallback_summary).strip()
    action_items: list[ActionItem] = []
    for item in parsed.get("action_items") or []:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task") or "").strip()
        if not task:
            continue
        owner_val = item.get("owner")
        due_val = item.get("due")
        action_items.append(
            ActionItem(
                task=task,
                owner=str(owner_val).strip() if owner_val and str(owner_val).lower() not in ("null", "none", "") else None,
                due=str(due_val).strip() if due_val and str(due_val).lower() not in ("null", "none", "") else None,
            )
        )
    return summary, action_items


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def summarize_meeting_text(text: str, *, use_llm: bool) -> MeetingSummaryResponse:
    cleaned = preprocess_transcript(text)
    if not cleaned:
        return MeetingSummaryResponse(
            summary="No text content found.",
            action_items=[],
            raw_transcript_length=0,
        )

    # Fallback summary (no LLM): first N chars
    fallback_summary = cleaned[:800] + ("..." if len(cleaned) > 800 else "")
    action_items: list[ActionItem] = []
    summary = fallback_summary

    if use_llm and settings.llm_enabled:
        max_chunk = settings.meeting_chunk_chars
        overlap = settings.meeting_chunk_overlap_chars
        chunks = chunk_transcript(cleaned, max_chars=max_chunk, overlap_chars=overlap)

        logger.info(
            "Meeting summarization: %d chars, %d chunk(s) (chunk_size=%d)",
            len(cleaned), len(chunks), max_chunk,
        )

        partial_results: list[dict] = []
        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            parsed = _call_with_retry(_CHUNK_SYSTEM, chunk)
            if parsed:
                partial_results.append(parsed)
            else:
                logger.warning("Chunk %d/%d produced no result", i + 1, len(chunks))

        if not partial_results:
            logger.error("All chunks failed — returning fallback summary")
        elif len(partial_results) == 1:
            # Single chunk — use directly
            summary, action_items = _parse_result(partial_results[0], fallback_summary)
        else:
            # Multiple chunks — merge with a second LLM call
            merge_input_parts = []
            for idx, pr in enumerate(partial_results, start=1):
                part_summary = pr.get("summary", "(no summary)")
                part_actions = pr.get("action_items", [])
                action_lines = "\n".join(
                    f"  - {a.get('task', '')} | owner: {a.get('owner', 'null')} | due: {a.get('due', 'null')}"
                    for a in part_actions if isinstance(a, dict) and a.get("task")
                ) or "  (none)"
                merge_input_parts.append(
                    f"=== Section {idx} Summary ===\n{part_summary}\n\n"
                    f"=== Section {idx} Action Items ===\n{action_lines}"
                )
            merge_input = "\n\n".join(merge_input_parts)

            logger.info("Merging %d partial results with second LLM call", len(partial_results))
            merged = _call_with_retry(_MERGE_SYSTEM, merge_input)
            if merged:
                summary, action_items = _parse_result(merged, fallback_summary)
            else:
                # Fallback: concatenate partial summaries
                logger.warning("Merge call failed — concatenating partial summaries")
                summary_parts = [pr.get("summary", "") for pr in partial_results if pr.get("summary")]
                summary = " ".join(summary_parts) if summary_parts else fallback_summary
                for pr in partial_results:
                    _, items = _parse_result(pr, "")
                    action_items.extend(items)
                # Basic dedup by task text
                seen: set[str] = set()
                unique_items: list[ActionItem] = []
                for item in action_items:
                    key = item.task.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                action_items = unique_items

    return MeetingSummaryResponse(
        summary=summary,
        action_items=action_items,
        raw_transcript_length=len(cleaned),
    )
