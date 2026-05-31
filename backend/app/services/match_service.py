import json
import logging
import re
from pathlib import Path

from app.config import settings
from app.models.schemas import MatchResponse, MatchedPair, RequirementMatch, SourceChunk
from app.services import auth_service, jd_extract, jd_normalize, jd_parser, llm, match_retrieval, rag, reranker, vector_store
from app.services import skill_ontology

logger = logging.getLogger(__name__)

_HEADER_BOILERPLATE_RE = re.compile(
    r"(linkedin\.com|github\.com|@[\w.+-]+\.|work rights|availability:|"
    r"email:|languages:|require sponsorship)",
    re.IGNORECASE,
)
_KEYWORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}")

# Tokens where substring match causes false positives (e.g. "java" inside "javascript").
_STRICT_BOUNDARY_TOKENS = frozenset(
    {"java", "api", "aws", "sql", "net", "go", "r", "c", "ai"}
)

# Generic JD words that must not alone justify a citation.
_CITATION_STOPWORDS = frozenset(
    {
        "with", "and", "the", "for", "including", "experience", "design",
        "implement", "strong", "understanding", "capabilities", "management",
        "products", "applications", "enterprise", "data", "documentation",
        "cloud", "services", "operate", "build", "related", "field",
        "ideally", "pipelines", "automation", "environments", "modelling", "workflows",
    }
)

# Education regex (generic — degrees appear in most JDs regardless of domain).
_EDUCATION_REQ_RE = re.compile(
    r"\b(degree|bachelor|master|b\.?s\.?c|b\.?eng|diploma|qualification|graduate)\b",
    re.IGNORECASE,
)
_DEGREE_RESUME_RE = re.compile(
    r"\b(bachelor|master|b\.?s\.?c|b\.?a\.?|b\.?eng|m\.?s\.?c|degree|graduate|diploma)\b",
    re.IGNORECASE,
)

# Stop-words for dynamic token extraction.
_TOKEN_STOPWORDS = frozenset({
    "with", "and", "the", "for", "using", "strong", "experience", "knowledge",
    "including", "management", "data", "cloud", "services", "build", "develop",
    "design", "implement", "work", "ability", "understanding", "skills", "solid",
    "good", "great", "excellent", "related", "field", "ideally", "preferred",
    "required", "minimum", "demonstrated", "proven", "working", "industry",
    "relevant", "communication", "team", "environment", "various", "other",
    "apply", "such", "area", "able", "both", "their", "have", "will", "also",
    "familiarity", "processing", "technologies", "technology", "development",
    "software", "applications", "systems", "processes", "concepts", "fundamentals",
    "principles", "methodologies", "methods", "practices", "techniques", "tools",
    "platforms", "frameworks", "languages", "libraries", "apis", "solutions",
    "exposure", "understanding", "expertise", "competency", "proficiency",
    "proficient", "capable", "highly", "key", "core", "general", "basic",
    "advanced", "expert", "various", "multiple", "different", "use", "used",
    "applying", "applied"
})


def _extract_key_tech_tokens(text: str) -> list[str]:
    """
    Dynamically extract meaningful technical tokens from any requirement text.
    Works for any technology domain — no hardcoded tech names.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{2,}", text)
    return [t for t in tokens if t.lower() not in _TOKEN_STOPWORDS and len(t) >= 3]


_HEADER_BOILERPLATE_RE = re.compile(
    r"(linkedin\.com|github\.com|@[\w.+-]+\.|work rights|availability:|"
    r"email:|languages:|require sponsorship)",
    re.IGNORECASE,
)
_KEYWORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}")

# Tokens where substring match causes false positives (e.g. java in javascript).
_STRICT_BOUNDARY_TOKENS = frozenset(
    {"java", "api", "aws", "sql", "net", "go", "r", "c", "ai"}
)

# Generic JD words that must not alone justify a citation.
_CITATION_STOPWORDS = frozenset(
    {
        "with", "and", "the", "for", "including", "experience", "design",
        "implement", "strong", "understanding", "capabilities", "management",
        "products", "applications", "enterprise", "data", "documentation",
        "cloud", "services", "operate", "build", "related", "field",
        "ideally", "pipelines", "automation", "environments", "modelling", "workflows",
    }
)

# Education regex (generic — degrees appear in most JDs regardless of domain).
_EDUCATION_REQ_RE = re.compile(
    r"\b(degree|bachelor|master|b\.?s\.?c|b\.?eng|diploma|qualification|graduate)\b",
    re.IGNORECASE,
)
_DEGREE_RESUME_RE = re.compile(
    r"\b(bachelor|master|b\.?s\.?c|b\.?a\.?|b\.?eng|m\.?s\.?c|degree|graduate|diploma)\b",
    re.IGNORECASE,
)




def _education_match_score(req_text: str, resume_text: str) -> float:
    """
    Generic degree/education requirement matching.
    Extracts field keywords dynamically — works for any degree field.
    """
    if not _EDUCATION_REQ_RE.search(req_text) and "related field" not in req_text.lower():
        return 0.0
    if not _DEGREE_RESUME_RE.search(resume_text):
        return 0.0

    req_lower = req_text.lower()
    resume_lower = resume_text.lower()
    has_related_clause = bool(
        re.search(r"\brelated field\b|\bor equivalent\b|\bequivalent qualification\b", req_text, re.I)
    )

    # Dynamically extract field/discipline keywords from requirement
    field_candidates = re.findall(r"[a-z]{5,}", req_lower)
    field_stopwords = {
        "degree", "bachelor", "master", "field", "related", "science",
        "engineering", "qualification", "graduate", "required", "minimum",
        "equivalent", "discipline", "study", "studies",
    }
    specific_fields = [w for w in field_candidates if w not in field_stopwords]

    for field in specific_fields:
        if field in resume_lower:
            return 0.58

    if has_related_clause and _DEGREE_RESUME_RE.search(resume_text):
        return 0.45

    # Has a degree but no matching field found
    return 0.20


def _best_education_chunk(
    resume_chunks: list[SourceChunk],
) -> SourceChunk | None:
    for ch in resume_chunks:
        if ch.section == "education":
            return ch
    for ch in resume_chunks:
        if _DEGREE_RESUME_RE.search(ch.text):
            return ch
    return None





def _enforce_technology_evidence(
    req_text: str,
    best: SourceChunk | None,
    relevance: float,
) -> float:
    """
    Generalised evidence enforcement using dynamic token extraction.
    If the requirement names specific technologies, check if those tokens appear
    in the resume evidence. Works for any domain — no hardcoded tech names.
    """
    if not best:
        return relevance
    key_tokens = _extract_key_tech_tokens(req_text)
    if not key_tokens:
        return relevance
    text = best.text
    matched = sum(
        1 for t in key_tokens
        if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE)
    )
    ratio = matched / len(key_tokens)
    if ratio >= 0.6:
        # Strong token overlap — ensure at least partial threshold
        return min(1.0, max(relevance, 0.45))
    elif ratio >= 0.35:
        # Moderate overlap — mild boost
        return min(1.0, relevance + 0.04)
    elif ratio < 0.15 and relevance >= 0.45:
        # Scored as met but almost no matching tokens — cap to partial
        return min(relevance, 0.40)
    return relevance



def _requirement_queries(req_text: str, *, category: str = "") -> list[str]:
    """
    Build retrieval queries for a requirement using ontology expansion.
    Relies on skill_ontology for domain-aware synonym expansion.
    No hardcoded technology-specific query appends.
    """
    queries = skill_ontology.expand_search_queries(req_text, category=category)
    if req_text not in queries:
        queries.insert(0, req_text)
    return list(dict.fromkeys(queries))


def _infer_target_section(req_text: str, category: str) -> str | None:
    cat = (category or "").lower()
    t = req_text.lower()
    if cat == "skill" or re.search(r"\b(python|javascript|typescript|react|api|sql)\b", t):
        return "skills"
    if cat == "experience" or re.search(r"\b(year|experience|production|deliver)\b", t):
        return "experience"
    if cat == "education":
        return "education"
    if re.search(r"\b(llm|generative|machine learning|ai)\b", t):
        return "projects"
    return None


def _is_header_boilerplate(text: str, chunk_index: int) -> bool:
    if chunk_index > 1:
        return False
    head = text[:500]
    return len(_HEADER_BOILERPLATE_RE.findall(head)) >= 2


def _keyword_overlap_boost(req_text: str, chunk_text: str, relevance: float) -> float:
    req_tokens = {t.lower() for t in _KEYWORD_RE.findall(req_text) if len(t) > 3}
    if not req_tokens:
        return relevance
    overlap = sum(1 for t in req_tokens if _token_in_text(t, chunk_text))
    if overlap >= 3:
        return min(1.0, relevance + 0.15)
    if overlap >= 2:
        return min(1.0, relevance + 0.10)
    if overlap == 1:
        return min(1.0, relevance + 0.05)
    return relevance


def _score_requirement_hit(
    hit: SourceChunk,
    *,
    req_text: str,
    category: str,
) -> float:
    rel = reranker.relevance_0_1(
        rerank_score=hit.rerank_score,
        vector_score=hit.vector_score,
    )
    rel = _keyword_overlap_boost(req_text, hit.text, rel)
    if _is_header_boilerplate(hit.text, hit.chunk_index):
        rel = max(0.0, rel - 0.15)
    target = _infer_target_section(req_text, category)
    if target and hit.section == target:
        rel = min(1.0, rel + 0.10)
    elif target and hit.section in ("general", "summary") and target != "summary":
        rel = max(0.0, rel - 0.05)
    return rel


def _apply_core_skill_floor(
    req_text: str,
    best: SourceChunk | None,
    relevance: float,
) -> float:
    """
    Generic skill floor: prevent false 'missing' when resume clearly contains
    key tokens from the requirement. Works for any technology domain.
    """
    if not best:
        return relevance
    key_tokens = _extract_key_tech_tokens(req_text)
    if not key_tokens:
        # Fall back to education matching if no tech tokens
        edu = _education_match_score(req_text, best.text)
        if edu >= 0.25:
            return max(relevance, edu)
        return relevance
    text = best.text
    matched = sum(
        1 for t in key_tokens
        if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE)
    )
    ratio = matched / len(key_tokens)
    if ratio >= 0.5:
        return max(relevance, 0.45)  # Strong match — ensure at least met threshold
    elif ratio >= 0.3:
        return max(relevance, 0.30)  # Partial evidence floor
    # Also check education requirements
    edu = _education_match_score(req_text, text)
    if edu >= 0.25:
        return max(relevance, edu)
    return relevance


def _refine_best_evidence(
    best: SourceChunk | None,
    all_hits: list[SourceChunk],
    *,
    req_text: str,
    category: str,
    relevance: float,
) -> tuple[SourceChunk | None, float]:
    """
    Find the evidence chunk with the best key-token coverage for the requirement.
    Uses dynamic token extraction — works for any JD domain.
    """
    if not all_hits:
        return best, relevance
    key_tokens = _extract_key_tech_tokens(req_text)
    if not key_tokens:
        # Fall back to score-based selection
        refined = best
        best_rel = relevance
        for h in all_hits:
            rel = _score_requirement_hit(h, req_text=req_text, category=category)
            rel = _apply_core_skill_floor(req_text, h, rel)
            if rel > best_rel:
                best_rel = rel
                refined = h
        return refined, best_rel

    def _token_count(chunk_text: str) -> int:
        return sum(
            1 for t in key_tokens
            if re.search(rf"\b{re.escape(t)}\b", chunk_text, re.IGNORECASE)
        )

    best_count = _token_count(best.text) if best else 0
    best_chunk = best
    best_rel = relevance

    for h in all_hits:
        count = _token_count(h.text)
        rel = _score_requirement_hit(h, req_text=req_text, category=category)
        rel = _apply_core_skill_floor(req_text, h, rel)
        if count > best_count or rel > best_rel:
            best_count = max(best_count, count)
            best_rel = max(best_rel, rel)
            best_chunk = h
    return best_chunk, best_rel





def _strengths_gaps_from_matches(
    matches: list[RequirementMatch],
) -> tuple[list[str], list[str]]:
    min_p = settings.match_strength_partial_min_score
    strengths = [r.requirement for r in matches if r.status == "met"]
    strengths.extend(
        r.requirement
        for r in matches
        if r.status == "partial" and r.score >= min_p
    )
    gaps = [r.requirement for r in matches if r.status == "missing"]
    gaps.extend(
        r.requirement
        for r in matches
        if r.status == "partial" and r.score < min_p
    )
    return strengths[:12], gaps[:12]


def _pick_best_hit(
    hits: list[SourceChunk],
    *,
    req_text: str,
    category: str,
) -> tuple[SourceChunk | None, float]:
    if not hits:
        return None, 0.0
    scored = [
        (_score_requirement_hit(h, req_text=req_text, category=category), h) for h in hits
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][0]


def _load_resume_chunks(*, user_id: str, document_id: str) -> list[SourceChunk]:
    where = vector_store.build_where_filter(
        user_id=user_id,
        document_id=document_id,
        filename=None,
        doc_type=None,
        page_min=None,
        page_max=None,
    )
    ids, docs, metas = vector_store.fetch_chunks(where, limit=120)
    chunks: list[SourceChunk] = []
    for i, text in enumerate(docs):
        if not text:
            continue
        meta = metas[i] or {}
        chunk_index = int(meta.get("chunk_index", i))
        chunks.append(
            SourceChunk(
                chunk_id=ids[i],
                document_id=str(meta.get("document_id", document_id)),
                filename=str(meta.get("filename", "")),
                page=int(meta.get("page", 1)),
                chunk_index=chunk_index,
                section=str(meta.get("section", "general")),
                text=text,
            )
        )
    return chunks


def _required_resume_pattern(req_text: str) -> re.Pattern | None:
    """
    Build a dynamic pattern from key tokens in the requirement text.
    Used for citation selection — finds the most relevant resume chunk.
    """
    key_tokens = _extract_key_tech_tokens(req_text)
    if not key_tokens:
        return None
    # Build a regex that matches any of the key tokens with word boundaries
    pattern = "|".join(rf"\b{re.escape(t)}\b" for t in key_tokens[:8])  # cap at 8 for performance
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None





def _select_evidence_chunk(
    req_text: str,
    best: SourceChunk | None,
    *,
    resume_chunks: list[SourceChunk],
    all_hits: list[SourceChunk],
) -> SourceChunk | None:
    """Pick a citation chunk that actually mentions the required technology."""
    pat = _required_resume_pattern(req_text)
    if not pat:
        return best

    candidates: list[SourceChunk] = []
    seen: set[str] = set()
    for ch in list(all_hits) + list(resume_chunks):
        if ch.chunk_id in seen:
            continue
        seen.add(ch.chunk_id)
        if not pat.search(ch.text):
            continue
        if _is_header_boilerplate(ch.text, ch.chunk_index):
            continue
        candidates.append(ch)

    if not candidates:
        return best

    def _rank(ch: SourceChunk) -> tuple[int, int, int]:
        section_bonus = 2 if ch.section in ("skills", "projects", "experience") else 0
        hit_count = len(pat.findall(ch.text))
        return (section_bonus, hit_count, len(ch.text))

    candidates.sort(key=_rank, reverse=True)
    return candidates[0]


def _resume_wide_skill_boost(
    req_text: str,
    *,
    best: SourceChunk | None,
    relevance: float,
    resume_chunks: list[SourceChunk],
) -> tuple[SourceChunk | None, float]:
    """
    Scan the full indexed resume for key requirement tokens.
    Dynamic extraction — works for any JD without a hardcoded floor table.
    """
    if not resume_chunks:
        return best, relevance
    key_tokens = _extract_key_tech_tokens(req_text)
    if not key_tokens:
        return best, relevance

    corpus = "\n".join(c.text for c in resume_chunks)
    corpus_matched = sum(
        1 for t in key_tokens
        if re.search(rf"\b{re.escape(t)}\b", corpus, re.IGNORECASE)
    )
    corpus_ratio = corpus_matched / len(key_tokens)

    if corpus_ratio < 0.25:
        # Key tokens mostly absent from the entire resume
        return best, relevance

    # Find the individual chunk with the highest token coverage
    def _chunk_token_count(ch: SourceChunk) -> int:
        return sum(
            1 for t in key_tokens
            if re.search(rf"\b{re.escape(t)}\b", ch.text, re.IGNORECASE)
        )

    best_by_token = max(resume_chunks, key=_chunk_token_count, default=None)
    if not best_by_token:
        return best, relevance
    chunk_count = _chunk_token_count(best_by_token)
    if chunk_count == 0:
        return best, relevance

    # Dynamic floor proportional to corpus coverage
    floor = min(0.28 + corpus_ratio * 0.28, 0.52)
    if floor > relevance:
        return best_by_token, floor
    return best, relevance




def _citation_overlaps_requirement(req_text: str, chunk_text: str) -> bool:
    req = req_text.lower()
    pat = _required_resume_pattern(req_text)
    if pat:
        return bool(pat.search(chunk_text))

    req_tokens = {
        t.lower()
        for t in _KEYWORD_RE.findall(req_text)
        if len(t) > 3 and t.lower() not in _CITATION_STOPWORDS
    }
    if not req_tokens:
        return True
    overlap = sum(1 for t in req_tokens if _token_in_text(t, chunk_text))
    if overlap >= 2:
        return True
    if overlap == 1 and len(req_tokens) <= 3:
        return True
    return False



def _build_explanation(
    *,
    req_text: str,
    status: str,
    relevance: float,
    best: SourceChunk | None,
) -> tuple[str, str | None]:
    pct = round(relevance * 100, 1)
    excerpt = best.text[:420].strip() if best and best.text.strip() else None
    req_short = req_text if len(req_text) <= 140 else req_text[:137] + "…"

    if status == "met":
        page = best.page if best else "?"
        expl = (
            f"Strong match ({pct}%). Resume text on page {page} closely aligns with this requirement."
        )
        pat = _required_resume_pattern(req_text)
        if pat and best and pat.search(best.text):
            expl += " Citation shows the matching skill or technology on your resume."
    elif status == "partial":
        page = best.page if best else "?"
        expl = (
            f"Partial match ({pct}%). Related resume content on page {page} was found; "
            f"wording may differ from the job description."
        )
        if _required_resume_pattern(req_text) and best:
            key_tokens = _extract_key_tech_tokens(req_text)
            missing = [
                t for t in key_tokens
                if not re.search(rf"\b{re.escape(t)}\b", best.text, re.IGNORECASE)
            ]
            if missing:
                expl += f" Resume does not mention: {', '.join(missing[:4])}."
    elif best and relevance >= 0.08:
        page = best.page
        section = getattr(best, "section", "general")
        expl = (
            f"Weak match ({pct}%). Closest resume passage (page {page}, section: {section}) "
            f"did not meet the match threshold (met ≥ 45%, partial ≥ 25%) for: {req_short}"
        )
    else:
        expl = (
            f"No match ({pct}%). No convincing resume evidence was found for: {req_short} "
            f"(thresholds: met ≥ 45%, partial ≥ 25%)."
        )
    return expl, excerpt


_SECTION_READ_ORDER = {
    "skills": 0,
    "experience": 1,
    "projects": 2,
    "education": 3,
    "summary": 4,
    "general": 5,
}


def _build_resume_context(resume_chunks: list[SourceChunk]) -> str:
    """Full resume text for LLM judge (not chunk-by-chunk retrieval)."""
    if not resume_chunks:
        return ""
    ordered = sorted(
        resume_chunks,
        key=lambda c: (
            _SECTION_READ_ORDER.get(c.section, 9),
            c.page,
            c.chunk_index,
        ),
    )
    parts: list[str] = []
    total = 0
    limit = settings.match_resume_context_max_chars
    for ch in ordered:
        block = ch.text.strip()
        if not block:
            continue
        if total + len(block) > limit:
            remain = limit - total
            if remain > 200:
                parts.append(block[:remain])
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


def _find_chunk_for_quote(
    quote: str | None,
    resume_chunks: list[SourceChunk],
) -> SourceChunk | None:
    if not quote or not resume_chunks:
        return None
    q = quote.strip().lower()
    if len(q) < 12:
        return None
    for ch in resume_chunks:
        if q in ch.text.lower():
            return ch
    # Prefix match on first 40 chars
    prefix = q[:40]
    for ch in resume_chunks:
        if prefix in ch.text.lower():
            return ch
    return None


def _sanitize_judge_row(
    req_text: str,
    resume_context: str,
    *,
    status: str,
    score: float,
    reason: str,
) -> tuple[str, float, str]:
    """Ground-truth fixes for common LLM confusions (GCP, MCP vs Google)."""
    req_l = req_text.lower()
    res_l = resume_context.lower()
    if re.search(r"\b(gcp|google cloud|vertex ai|cloud run|cloud sql)\b", req_l):
        has_gcp = bool(
            re.search(
                r"\b(gcp|google cloud|vertex(\s+ai)?|cloud run|cloud sql)\b",
                res_l,
                re.I,
            )
        )
        if not has_gcp:
            return (
                "missing",
                min(score, 22.0),
                "Resume does not mention Google Cloud (GCP), Vertex AI, Cloud Run, or Cloud SQL.",
            )

    if re.search(r"\bmcp\b|model context protocol", req_l, re.I):
        reason_l = reason.lower()
        if reason and re.search(r"\bgcp\b|vertex|google cloud", reason_l):
            if not re.search(r"\bmcp\b|model context protocol", reason_l):
                reason = (
                    "MCP (Model Context Protocol) is not the same as GCP or Vertex AI. "
                    "Assess only whether the resume shows MCP or equivalent tool-exposure patterns."
                )
    return status, score, reason


def _llm_judge_requirements_batch(
    *,
    resume_context: str,
    requirements: list[dict],
    resume_chunks: list[SourceChunk],
) -> list[RequirementMatch] | None:
    """LLM reads full resume and judges each requirement semantically."""
    if not resume_context or not requirements:
        return None

    indexed = [
        {"index": i, "text": r["text"], "category": r.get("category", "other")}
        for i, r in enumerate(requirements)
    ]
    raw = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You compare a RESUME to JOB REQUIREMENTS one row at a time. "
                    "Use semantic judgment: equivalent experience counts (e.g. Chroma + "
                    "hybrid BM25+vector + embeddings for 'vector search' or RAG in another stack; "
                    "not only MySQL vector). "
                    "Hard rules: "
                    "(1) Model Context Protocol (MCP) is NOT Google Cloud Platform (GCP). "
                    "Never explain MCP using GCP/Vertex. MCP is an AI tool-calling protocol; "
                    "GCP is cloud infrastructure. Judge them separately. "
                    "(2) For GCP/Vertex AI/Cloud Run: use met or partial only if the resume "
                    "mentions GCP, Google Cloud, Vertex, Cloud Run, or another named Google "
                    "cloud product. Do not infer GCP from unrelated tools (Netlify, AWS alone, Ollama). "
                    "If there is no such mention, status must be missing. "
                    "(3) Every reason and evidence must be grounded in text from the resume. "
                    "Do not claim the candidate has a skill unless the resume supports it. "
                    "Quote or paraphrase concrete resume phrases in evidence when possible. "
                    "Status: met = clearly demonstrated; partial = related but incomplete; "
                    "missing = no real evidence. "
                    "Score 0-100 = confidence. Use missing with score 0-24 when there is no evidence. "
                    'Respond with ONLY valid JSON: '
                    '{"matches": [{"index": number, "status": "met"|"partial"|"missing", '
                    '"score": number, "evidence": string|null, "reason": string}]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"RESUME:\n{resume_context}\n\n"
                    f"REQUIREMENTS:\n{json.dumps(indexed, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0.0,
    )
    parsed = llm.extract_json_block(raw or "")
    if not parsed:
        return None

    by_index: dict[int, dict] = {}
    for row in parsed.get("matches") or []:
        if isinstance(row, dict) and isinstance(row.get("index"), int):
            by_index[row["index"]] = row

    out: list[RequirementMatch] = []
    for i, req in enumerate(requirements):
        row = by_index.get(i, {})
        status = str(row.get("status") or "missing").lower()
        if status not in ("met", "partial", "missing"):
            status = "missing"
        score = float(row.get("score") or 0)
        score = max(0.0, min(100.0, score))
        if status == "met" and score < 45:
            score = 45.0
        elif status == "partial" and score < 25:
            score = 25.0
        elif status == "missing":
            score = min(score, 24.0)

        reason = str(row.get("reason") or "").strip()
        status, score, reason = _sanitize_judge_row(
            req["text"],
            resume_context,
            status=status,
            score=score,
            reason=reason,
        )

        evidence = row.get("evidence")
        if isinstance(evidence, str):
            evidence = evidence.strip() or None
        else:
            evidence = None

        chunk = _find_chunk_for_quote(evidence, resume_chunks)
        citation = None
        excerpt = evidence
        if chunk:
            citation = f"(p.{chunk.page}) {chunk.text[:280]}"
            if not excerpt:
                excerpt = chunk.text[:420]

        if reason:
            explanation = f"{reason} ({score}% confidence)."
        else:
            explanation = _build_explanation(
                req_text=req["text"],
                status=status,
                relevance=score / 100.0,
                best=chunk,
            )[0]

        out.append(
            RequirementMatch(
                requirement=req["text"],
                category=req.get("category", "other"),
                priority=req.get("priority", "required"),
                status=status,
                resume_citation=citation,
                resume_excerpt=excerpt,
                explanation=explanation,
                score=round(score, 1),
            )
        )
    return out


def _match_requirements_with_llm_judge(
    *,
    user_id: str,
    resume_document_id: str,
    requirements: list[dict],
    resume_chunks: list[SourceChunk],
) -> list[RequirementMatch]:
    resume_context = _build_resume_context(resume_chunks)
    batch_size = max(1, settings.match_judge_batch_size)
    all_matches: list[RequirementMatch] = []

    for start in range(0, len(requirements), batch_size):
        batch = requirements[start : start + batch_size]
        judged = _llm_judge_requirements_batch(
            resume_context=resume_context,
            requirements=batch,
            resume_chunks=resume_chunks,
        )
        if judged and len(judged) == len(batch):
            all_matches.extend(judged)
            continue
        for req in batch:
            all_matches.append(
                _match_requirement(
                    user_id=user_id,
                    resume_document_id=resume_document_id,
                    requirement=req,
                    resume_chunks=resume_chunks,
                )
            )
    return all_matches


def _jd_query_text(jd_sources: list[SourceChunk], jd_plain: str | None = None) -> str:
    if jd_plain and jd_plain.strip():
        return jd_plain.strip()[:4000]
    if not jd_sources:
        return "job requirements skills qualifications experience responsibilities"
    return "\n".join(s.text for s in jd_sources[:5])[:2500]


def _match_requirement(
    *,
    user_id: str,
    resume_document_id: str,
    requirement: dict,
    resume_chunks: list[SourceChunk] | None = None,
) -> RequirementMatch:
    if settings.match_scoring_mode == "retrieval" and not settings.match_use_llm_judge:
        return match_retrieval.match_requirement(
            user_id=user_id,
            resume_document_id=resume_document_id,
            requirement=requirement,
            resume_chunks=resume_chunks,
        )

    req_text = requirement["text"]
    category = str(requirement.get("category") or "other")
    best = None
    relevance = 0.0
    seen_ids: set[str] = set()
    all_hits: list[SourceChunk] = []

    for q in _requirement_queries(req_text, category=category):
        hits = rag.retrieve_and_rerank(
            user_id=user_id,
            question=q,
            document_id=resume_document_id,
            filename=None,
            doc_type=None,
            page_min=None,
            page_max=None,
            top_k=settings.match_requirement_top_k,
            do_rerank=settings.rerank_enabled,
            expand_pages=False,
        )
        for hit in hits:
            if hit.chunk_id not in seen_ids:
                seen_ids.add(hit.chunk_id)
                all_hits.append(hit)

    best, relevance = _pick_best_hit(all_hits, req_text=req_text, category=category)
    best, relevance = _refine_best_evidence(
        best,
        all_hits,
        req_text=req_text,
        category=category,
        relevance=relevance,
    )
    relevance = _apply_core_skill_floor(req_text, best, relevance)

    if resume_chunks:
        best, relevance = _resume_wide_skill_boost(
            req_text,
            best=best,
            relevance=relevance,
            resume_chunks=resume_chunks,
        )

    relevance = _enforce_technology_evidence(req_text, best, relevance)

    if category == "education" or _EDUCATION_REQ_RE.search(req_text):
        corpus = "\n".join(c.text for c in resume_chunks) if resume_chunks else (best.text if best else "")
        edu_score = _education_match_score(req_text, corpus)
        if edu_score > relevance:
            relevance = edu_score
            edu_chunk = _best_education_chunk(resume_chunks) if resume_chunks else best
            if edu_chunk:
                best = edu_chunk

    # Generic web API token check (replaces hardcoded _API_RESUME_RE)
    if best and re.search(r"web\s*apis?", req_text, re.I):
        api_pat = re.compile(r"\b(rest|api|apis|endpoint|fastapi|http|graphql|webhook)\b", re.I)
        if api_pat.search(best.text):
            relevance = max(relevance, 0.42)

    evidence = _select_evidence_chunk(
        req_text,
        best,
        resume_chunks=resume_chunks or [],
        all_hits=all_hits,
    )
    if evidence:
        best = evidence

    req_pat = _required_resume_pattern(req_text)
    if req_pat and relevance >= 0.25:
        if not best or not req_pat.search(best.text):
            relevance = min(relevance, 0.22)

    if relevance >= 0.45:
        status = "met"
    elif relevance >= 0.25:
        status = "partial"
    else:
        status = "missing"

    citation = None
    if best and status != "missing" and _citation_overlaps_requirement(req_text, best.text):
        citation = f"(p.{best.page}) {best.text[:280]}"
    elif best and status != "missing":
        citation = None
        if status == "met":
            status = "partial"
        elif status == "partial" and relevance < 0.30:
            status = "missing"

    explanation, excerpt = _build_explanation(
        req_text=req_text,
        status=status,
        relevance=relevance,
        best=best,
    )

    return RequirementMatch(
        requirement=req_text,
        category=requirement.get("category", "other"),
        priority=requirement.get("priority", "required"),
        status=status,
        resume_citation=citation,
        resume_excerpt=excerpt,
        explanation=explanation,
        score=round(relevance * 100, 1),
    )


def _get_or_ingest_resume(
    *,
    user_id: str,
    resume_path: Path,
    resume_filename: str,
    content_hash: str,
) -> str:
    if settings.match_reuse_indexed_docs:
        existing = auth_service.find_document_by_hash(
            user_id=user_id, doc_type="resume", content_hash=content_hash
        )
        if existing:
            return existing

    resume_id, resume_chunks = rag.ingest_document_file(
        file_path=resume_path,
        original_filename=resume_filename,
        user_id=user_id,
        doc_type="resume",
    )
    auth_service.register_document(
        user_id=user_id,
        document_id=resume_id,
        filename=resume_filename,
        doc_type="resume",
        chunks=resume_chunks,
        content_hash=content_hash,
    )
    return resume_id


def _get_or_ingest_jd(
    *,
    user_id: str,
    jd_path: Path | None,
    jd_filename: str,
    jd_plain: str | None,
    content_hash: str,
) -> str:
    if settings.match_reuse_indexed_docs:
        existing = auth_service.find_document_by_hash(
            user_id=user_id, doc_type="jd", content_hash=content_hash
        )
        if existing:
            return existing

    if jd_plain:
        jd_id, jd_chunks = rag.ingest_plain_text(
            text=jd_plain,
            original_filename=jd_filename if jd_filename.endswith(".txt") else "job_description.txt",
            user_id=user_id,
            doc_type="jd",
        )
    else:
        assert jd_path is not None
        jd_id, jd_chunks = rag.ingest_pdf_file(
            file_path=jd_path,
            original_filename=jd_filename,
            user_id=user_id,
            doc_type="jd",
        )

    auth_service.register_document(
        user_id=user_id,
        document_id=jd_id,
        filename=jd_filename,
        doc_type="jd",
        chunks=jd_chunks,
        content_hash=content_hash,
    )
    return jd_id


def match_resume_to_jd(
    *,
    user_id: str,
    resume_path: Path,
    resume_filename: str,
    resume_content_hash: str,
    jd_path: Path | None = None,
    jd_filename: str = "job_description.txt",
    jd_text: str | None = None,
    jd_content_hash: str,
    use_llm: bool,
) -> MatchResponse:
    if not jd_path and not (jd_text and jd_text.strip()):
        raise ValueError("Provide JD as pasted text or PDF")

    resume_id = _get_or_ingest_resume(
        user_id=user_id,
        resume_path=resume_path,
        resume_filename=resume_filename,
        content_hash=resume_content_hash,
    )

    jd_plain = jd_text.strip() if jd_text else None
    jd_id = _get_or_ingest_jd(
        user_id=user_id,
        jd_path=jd_path,
        jd_filename=jd_filename,
        jd_plain=jd_plain,
        content_hash=jd_content_hash,
    )

    jd_full_text = jd_plain or ""
    if not jd_full_text and jd_path:
        from app.services.document_extract import extract_text_from_file

        jd_full_text = extract_text_from_file(jd_path, jd_filename)

    jd_sources = rag.retrieve_and_rerank(
        user_id=user_id,
        question="required skills qualifications experience responsibilities education",
        document_id=jd_id,
        filename=None,
        doc_type=None,
        page_min=None,
        page_max=None,
        top_k=6,
        do_rerank=settings.rerank_enabled,
        expand_pages=False,
    )

    requirements = jd_normalize.enrich_requirements(jd_parser.parse_requirements(jd_full_text))
    cover_letter_topics = jd_parser.parse_cover_letter_traits(jd_full_text)
    resume_chunks = _load_resume_chunks(user_id=user_id, document_id=resume_id)
    scoring_mode = (settings.match_scoring_mode or "retrieval").strip().lower()
    use_judge = (
        scoring_mode == "llm_judge"
        and use_llm
        and settings.llm_enabled
        and settings.match_use_llm_judge
        and resume_chunks
    )
    if use_judge:
        requirement_matches = _match_requirements_with_llm_judge(
            user_id=user_id,
            resume_document_id=resume_id,
            requirements=requirements,
            resume_chunks=resume_chunks,
        )
    else:
        requirement_matches = [
            match_retrieval.match_requirement(
                user_id=user_id,
                resume_document_id=resume_id,
                requirement=req,
                resume_chunks=resume_chunks,
            )
            for req in requirements
        ]

    jd_query = _jd_query_text(jd_sources, jd_plain)
    resume_sources = rag.retrieve_and_rerank(
        user_id=user_id,
        question=jd_query,
        document_id=resume_id,
        filename=None,
        doc_type=None,
        page_min=None,
        page_max=None,
        top_k=12,
        do_rerank=settings.rerank_enabled,
        expand_pages=False,
    )

    pairs: list[MatchedPair] = []
    for rs in resume_sources[:5]:
        pairs.append(
            MatchedPair(
                jd_excerpt=(jd_sources[0].text[:400] if jd_sources else jd_full_text[:400]),
                resume_excerpt=rs.text[:500],
                relevance_score=reranker.relevance_0_1(
                    rerank_score=rs.rerank_score,
                    vector_score=rs.vector_score,
                ),
            )
        )

    requirement_matches.sort(key=lambda r: r.requirement.lower())

    if requirement_matches:
        required = [r for r in requirement_matches if r.priority != "preferred"]
        preferred = [r for r in requirement_matches if r.priority == "preferred"]
        scored = required if required else requirement_matches

        def _weighted_req_score(r: RequirementMatch) -> float:
            """Status-based penalty so gaps genuinely lower the score."""
            if r.status == "met":
                return r.score
            elif r.status == "partial":
                return r.score * 0.65  # partial counts for 65% of raw score
            else:  # missing
                return min(r.score * 0.25, 15.0)  # missing capped at 15 points

        base = sum(_weighted_req_score(r) for r in scored) / max(len(scored), 1)
        if preferred:
            pref_avg = sum(_weighted_req_score(r) for r in preferred) / len(preferred)
            base = base * 0.90 + pref_avg * 0.10
        overall_score = round(min(100.0, base), 1)
    else:
        overall_score = 0.0
        if jd_full_text.strip():
            logger.warning(
                "No scorable requirements parsed from JD; score is 0 until parser finds "
                "Required Skills / Technical Stack lines."
            )

    strengths, gaps = _strengths_gaps_from_matches(requirement_matches)
    summary = None
    req_count = len(requirement_matches)
    trait_n = len(cover_letter_topics)
    if req_count == 0:
        score_note = (
            "Could not parse technical requirements from this JD format (need a skills section "
            "such as Required Skills, Key skills & experience, or Technical Stack). "
            "Re-paste the JD or restart the backend after update. "
            "Score is 0% — not a reflection of your resume."
        )
    elif use_judge:
        score_note = (
            f"Score uses {req_count} requirements judged by LLM against your full resume "
            f"(semantic match — equivalent skills count, not keyword-only). "
            f"{trait_n} cover-letter topic(s) excluded from %."
        )
    else:
        n_required = sum(1 for r in requirement_matches if r.priority != "preferred")
        n_preferred = sum(1 for r in requirement_matches if r.priority == "preferred")
        score_note = (
            f"Score uses {req_count} requirements ({n_required} required, {n_preferred} preferred): "
            f"overall % is the average of required item scores"
            f"{'' if n_preferred == 0 else ' (+ small preferred weight)'}. "
            f"{trait_n} cover-letter topic(s) excluded."
        )

    if use_llm and settings.llm_enabled:
        req_lines = "\n".join(
            f"- [{r.status}] ({r.score}%) {r.requirement}" for r in requirement_matches[:25]
        )
        raw = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You write a SHORT hiring comparison (2–4 sentences). "
                        "CRITICAL: You may ONLY cite skills as present if the REQUIREMENT TABLE "
                        "shows met or partial for that row. If a row is missing, you must NOT say "
                        "the candidate has that skill or cloud experience. "
                        "Never contradict the table (e.g. do not claim GCP experience if GCP is missing). "
                        "Do NOT conflate MCP with Google Cloud — they are different. "
                        "Do not invent fractions like '22/30'; the score below is authoritative. "
                        f"Official match score: {overall_score}% (average of required item scores). "
                        f"Do NOT say 100% or 'highly qualified' unless most rows are met with high scores (≥70%). "
                        "If requirement count is 0, say the JD could not be scored. "
                        'Respond with ONLY valid JSON: {"summary": string}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Official score: {overall_score}% ({req_count} requirements).\n"
                        "REQUIREMENT TABLE (SOURCE OF TRUTH — do not contradict):\n"
                        f"{req_lines}\n\n"
                        "Cover-letter topics (not scored): "
                        f"{'; '.join(cover_letter_topics[:8]) or 'none'}\n\n"
                        "Brief JD context (do not override the table):\n"
                        f"{jd_full_text[:1500]}\n"
                    ),
                },
            ],
            temperature=0.0,
        )
        parsed = llm.extract_json_block(raw or "")
        if parsed:
            summary = parsed.get("summary")
    if not summary and requirement_matches:
        met_n = sum(1 for r in requirement_matches if r.status == "met")
        partial_n = sum(1 for r in requirement_matches if r.status == "partial")
        summary = (
            f"Matched {met_n} met and {partial_n} partial of {len(requirement_matches)} "
            f"requirements (score {overall_score}%)."
        )

    return MatchResponse(
        resume_document_id=resume_id,
        jd_document_id=jd_id,
        overall_score=overall_score,
        score_note=score_note,
        summary=summary,
        strengths=strengths,
        gaps=gaps,
        cover_letter_topics=cover_letter_topics,
        matched_pairs=pairs,
        requirement_matches=requirement_matches,
        resume_sources=resume_sources,
        jd_sources=jd_sources,
    )
