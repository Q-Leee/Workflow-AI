import json
import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def chat(
    messages: list[dict],
    *,
    timeout: float = 180.0,
    temperature: float | None = None,
    model: str | None = None,
) -> str | None:
    """
    Send a chat request to Ollama.

    Args:
        messages:    OpenAI-style message list.
        timeout:     HTTP request timeout in seconds.
        temperature: Override the global temperature. None → use settings.ollama_temperature.
        model:       Override the model. None → use settings.ollama_chat_model.
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    temp = settings.ollama_temperature if temperature is None else temperature
    chosen_model = model or settings.ollama_chat_model
    payload = {
        "model": chosen_model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": temp},
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            if r.status_code >= 400:
                logger.error("Ollama chat %s [model=%s]: %s", r.status_code, chosen_model, r.text[:500])
                return None
            data = r.json()
            msg = (data.get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    except Exception:
        logger.exception("Ollama chat failed (model=%s)", chosen_model)
    return None


def meeting_chat(
    messages: list[dict],
    *,
    timeout: float = 240.0,
    temperature: float | None = None,
) -> str | None:
    """
    Chat call dedicated to meeting summarization.
    Uses MEETING_CHAT_MODEL if set, otherwise falls back to the global model.
    """
    model = settings.meeting_chat_model.strip() or None
    temp = settings.meeting_temperature if temperature is None else temperature
    return chat(messages, timeout=timeout, temperature=temp, model=model)


def extract_json_block(text: str) -> dict | None:
    """
    Extract a JSON object from LLM output.
    Handles:
      - Raw JSON (most common with temperature=0)
      - Fenced JSON blocks (```json … ```)
      - JSON embedded in surrounding prose
    """
    if not text:
        return None

    # 1. Fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 2. First { … last } (handles extra prose around JSON)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None
