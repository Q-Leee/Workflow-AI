import re
from dataclasses import dataclass

from app.config import settings

_SECTION_RE = re.compile(
    r"^("
    r"technical experience|professional experience|work experience|employment|experience|"
    r"professional summary|summary|profile|"
    r"education|academic|"
    r"skills|technical skills|programming|"
    r"projects|certifications|additional experience"
    r")\s*[:\-–—]?\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextChunk:
    page_start: int
    chunk_index: int
    text: str
    section: str = "general"


def _normalize_section(header_match: str) -> str:
    h = header_match.strip().lower().replace(":", "")
    if "skill" in h:
        return "skills"
    if "education" in h or "academic" in h:
        return "education"
    if "experience" in h or "employment" in h:
        return "experience"
    if "summary" in h or "profile" in h:
        return "summary"
    if "project" in h:
        return "projects"
    if "certif" in h:
        return "certifications"
    return h[:40] or "general"


_EXPERIENCE_JOB_START = re.compile(
    r"^("
    r"(?:[A-Z][\w\s/&.-]{2,60}\s+(?:at|@|\||–|—|-)\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r"|\d{4}\s*[-–—]\s*(?:\d{4}|Present|Current)"
    r")",
    re.IGNORECASE,
)


def _group_experience_blocks(paragraphs: list[str], section: str) -> list[str]:
    """Keep bullets under one role together before size-based splitting."""
    if section != "experience" or len(paragraphs) < 2:
        return paragraphs
    grouped: list[str] = []
    buf: list[str] = []
    max_block = max(settings.chunk_size, 1100)

    def flush() -> None:
        if buf:
            grouped.append("\n".join(buf))
            buf.clear()

    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        is_new_role = bool(lines and _EXPERIENCE_JOB_START.match(lines[0]))
        if is_new_role and buf:
            flush()
        candidate = "\n".join(buf + [para]) if buf else para
        if buf and len(candidate) > max_block:
            flush()
            buf.append(para)
        else:
            buf.append(para)
    flush()
    return grouped if grouped else paragraphs


def chunk_page_text(
    page_number: int,
    text: str,
    chunk_index_start: int,
    section: str = "general",
) -> tuple[list[TextChunk], int]:
    """Split text by paragraphs first, then by size within long paragraphs."""
    raw = text.replace("\r\n", "\n")
    # Treat bullet lines as paragraph breaks (common in resumes)
    raw = re.sub(r"\n(?=[•●▪\-\*]\s)", "\n\n", raw)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs:
        paragraphs = [" ".join(raw.split())] if raw.strip() else []

    chunks: list[TextChunk] = []
    idx = chunk_index_start
    current_section = section

    for para in paragraphs:
        lines = para.split("\n")
        block_parts: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = _SECTION_RE.match(line)
            if m and len(line) < 80:
                current_section = _normalize_section(m.group(1))
                if len(line) > len(m.group(0)) + 2:
                    block_parts.append(line[m.end() :].strip())
                continue
            block_parts.append(line)
        block = " ".join(block_parts).strip()
        if not block:
            continue

        sub_blocks = [block]
        if current_section == "experience" and "\n" in block:
            sub_blocks = _group_experience_blocks(
                [p.strip() for p in block.split("\n") if p.strip()],
                current_section,
            )

        for block in sub_blocks:
            block = block.strip()
            if not block:
                continue

            size = settings.chunk_size
            overlap = min(settings.chunk_overlap, size - 1) if size > 1 else 0
            start = 0
            while start < len(block):
                end = min(len(block), start + size)
                piece = block[start:end].strip()
                if piece:
                    chunks.append(
                        TextChunk(
                            page_start=page_number,
                            chunk_index=idx,
                            text=piece,
                            section=current_section,
                        )
                    )
                    idx += 1
                if end >= len(block):
                    break
                start = max(0, end - overlap)

    return chunks, idx


def chunk_pdf_pages(pages: list[tuple[int, str]]) -> list[TextChunk]:
    out: list[TextChunk] = []
    next_idx = 0
    for page_no, text in pages:
        page_chunks, next_idx = chunk_page_text(page_no, text, next_idx)
        out.extend(page_chunks)
    return out
