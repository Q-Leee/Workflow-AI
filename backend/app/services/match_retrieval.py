"""
Retrieval-first requirement matching: ontology query expansion → wide recall → rerank → evidence spans.

Scores come from retrieval + rerank + evidence aggregation; LLM is not used here.
"""
from __future__ import annotations

import logging
import re

from app.config import settings
from app.models.schemas import RequirementMatch, SourceChunk
from app.services import evidence_spans, rag, requirement_knowledge, reranker, skill_ontology

logger = logging.getLogger(__name__)

_MET_THRESHOLD = 0.45
_PARTIAL_THRESHOLD = 0.25


def _status_from_relevance(rel: float) -> str:
    if rel >= _MET_THRESHOLD:
        return "met"
    if rel >= _PARTIAL_THRESHOLD:
        return "partial"
    return "missing"


def _score_hit(
    hit: SourceChunk,
    *,
    req_text: str,
    category: str,
) -> float:
    rel = reranker.relevance_0_1(
        rerank_score=hit.rerank_score,
        vector_score=hit.vector_score,
    )
    ont = skill_ontology.ontology_overlap_score(req_text, hit.text)
    rel = min(1.0, rel + 0.12 * ont)
    target = _infer_target_section(req_text, category)
    if target and hit.section == target:
        rel = min(1.0, rel + 0.08)
    return rel


def _infer_target_section(req_text: str, category: str) -> str | None:
    cat = (category or "").lower()
    t = req_text.lower()
    if cat == "skill" or re.search(r"\b(python|javascript|react|api|sql)\b", t):
        return "skills"
    if cat == "experience" or re.search(r"\b(year|experience|production)\b", t):
        return "experience"
    if cat == "education":
        return "education"
    if re.search(r"\b(llm|generative|machine learning|rag)\b", t):
        return "projects"
    return None


def _retrieve_pool(
    *,
    user_id: str,
    resume_document_id: str,
    queries: list[str],
    top_k: int,
) -> list[SourceChunk]:
    seen: set[str] = set()
    pool: list[SourceChunk] = []
    for q in queries:
        try:
            hits = rag.retrieve_and_rerank(
                user_id=user_id,
                question=q,
                document_id=resume_document_id,
                filename=None,
                doc_type=None,
                page_min=None,
                page_max=None,
                top_k=top_k,
                do_rerank=settings.rerank_enabled,
                expand_pages=False,
            )
        except Exception:
            logger.exception("Retrieval failed for query=%s", q[:80])
            continue
        for hit in hits:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                pool.append(hit)
    return pool


def _resume_wide_ontology_boost(
    req_text: str,
    *,
    best: SourceChunk | None,
    relevance: float,
    resume_chunks: list[SourceChunk],
) -> tuple[SourceChunk | None, float]:
    if not resume_chunks:
        return best, relevance
    corpus = "\n".join(c.text for c in resume_chunks)
    ont = skill_ontology.ontology_overlap_score(req_text, corpus)
    if ont < 0.45:
        return best, relevance

    best_chunk = best
    best_rel = relevance
    for ch in resume_chunks:
        ch_ont = skill_ontology.ontology_overlap_score(req_text, ch.text)
        rel = min(1.0, relevance + 0.10 * ch_ont)
        if rel > best_rel:
            best_rel = rel
            best_chunk = ch
    return best_chunk, best_rel


def _calibrate_score(
    req_text: str,
    best: SourceChunk | None,
    relevance: float,
    *,
    resume_corpus: str,
    knowledge_floor: float = 0.0,
) -> float:
    """
    Prevent weak retrieval + broad ontology from hitting met — but never below
    corpus knowledge floor (education / skill equivalence layer).
    """
    if knowledge_floor >= _MET_THRESHOLD:
        return max(relevance, knowledge_floor)

    if not best:
        return max(relevance, knowledge_floor)

    retrieval_core = reranker.relevance_0_1(
        rerank_score=best.rerank_score,
        vector_score=best.vector_score,
    )
    ont_best = skill_ontology.ontology_overlap_score(req_text, best.text)
    ont_corpus = skill_ontology.ontology_overlap_score(req_text, resume_corpus)

    if ont_corpus >= 0.4 and retrieval_core < 0.32 and relevance >= _MET_THRESHOLD:
        return max(min(relevance, 0.42), knowledge_floor)

    if retrieval_core < 0.25 and ont_best < 0.25:
        return max(min(relevance, 0.22), knowledge_floor)

    return max(relevance, knowledge_floor)


def _build_explanation(
    *,
    req_text: str,
    status: str,
    relevance: float,
    best: SourceChunk | None,
    concepts: list[str],
) -> tuple[str, str | None]:
    pct = round(relevance * 100, 1)
    excerpt = None
    if best and best.text.strip():
        excerpt = evidence_spans.extract_best_evidence(best.text, req_text)
    req_short = req_text if len(req_text) <= 140 else req_text[:137] + "…"
    concept_note = ""
    if concepts and status in ("met", "partial"):
        concept_note = f" Matched via related concepts: {', '.join(concepts[:4])}."

    if status == "met":
        page = best.page if best else "?"
        expl = (
            f"Strong match ({pct}%). Resume evidence on page {page} aligns with this requirement."
            f"{concept_note}"
        )
    elif status == "partial":
        page = best.page if best else "?"
        expl = (
            f"Partial match ({pct}%). Related experience on page {page}; "
            f"wording may differ from the JD.{concept_note}"
        )
    elif best and relevance >= 0.08:
        expl = (
            f"Weak match ({pct}%). Closest passage did not meet threshold "
            f"(met ≥ {_MET_THRESHOLD*100:.0f}%, partial ≥ {_PARTIAL_THRESHOLD*100:.0f}%) "
            f"for: {req_short}"
        )
    else:
        expl = (
            f"No match ({pct}%). No convincing resume evidence for: {req_short} "
            f"(retrieval + rerank + ontology; met ≥ {_MET_THRESHOLD*100:.0f}%)."
        )
    return expl, excerpt


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
    # Extract technical tokens dynamically
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{2,}", req_text)
    stopwords = {
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
    }
    key_tokens = [t for t in tokens if t.lower() not in stopwords and len(t) >= 3]
    if not key_tokens:
        return relevance
    text = best.text
    matched = sum(
        1 for t in key_tokens
        if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE)
    )
    ratio = matched / len(key_tokens)
    if ratio >= 0.6:
        # Strong match — ensure at least 48% (MET!)
        return min(1.0, max(relevance, 0.48))
    elif ratio >= 0.35:
        # Moderate match — mild boost
        return min(1.0, relevance + 0.04)
    elif ratio < 0.15 and relevance >= _MET_THRESHOLD:
        # Capped to partial if no matching tokens found in best chunk
        return min(relevance, 0.40)
    return relevance


def match_requirement(
    *,
    user_id: str,
    resume_document_id: str,
    requirement: dict,
    resume_chunks: list[SourceChunk] | None = None,
) -> RequirementMatch:
    req_text = str(requirement["text"])
    category = str(requirement.get("category") or "other")
    priority = str(requirement.get("priority") or "required")
    queries = requirement.get("search_queries")
    if not isinstance(queries, list) or not queries:
        queries = skill_ontology.expand_search_queries(req_text, category=category)

    fetch_k = max(settings.match_requirement_top_k, settings.match_retrieval_stage1_top_k)
    pool = _retrieve_pool(
        user_id=user_id,
        resume_document_id=resume_document_id,
        queries=queries,
        top_k=fetch_k,
    )

    best: SourceChunk | None = None
    relevance = 0.0
    if pool:
        scored = [(_score_hit(h, req_text=req_text, category=category), h) for h in pool]
        scored.sort(key=lambda x: x[0], reverse=True)
        best, relevance = scored[0][1], scored[0][0]

    if resume_chunks:
        best, relevance = _resume_wide_ontology_boost(
            req_text, best=best, relevance=relevance, resume_chunks=resume_chunks
        )

    corpus = "\n".join(c.text for c in resume_chunks) if resume_chunks else ""

    edu = requirement_knowledge.education_match_score(req_text, corpus)
    skill_floor = requirement_knowledge.corpus_skill_floor(req_text, corpus)
    knowledge_floor = max(edu, skill_floor)

    if knowledge_floor > 0:
        relevance = max(relevance, knowledge_floor)
        evidence = requirement_knowledge.find_skill_evidence_chunk(req_text, resume_chunks or [])
        if evidence is not None:
            best = evidence  # type: ignore[assignment]
        elif edu >= 0.45 and resume_chunks and not best:
            for ch in resume_chunks:
                if ch.section == "education" or re.search(
                    r"\b(degree|bachelor|engineering|university)\b", ch.text, re.I
                ):
                    best = ch
                    break

    relevance = _enforce_technology_evidence(req_text, best, relevance)
    relevance = _calibrate_score(
        req_text, best, relevance, resume_corpus=corpus, knowledge_floor=knowledge_floor
    )

    concepts = skill_ontology.concept_labels(req_text)
    status = _status_from_relevance(relevance)

    citation = None
    if best and status in ("met", "partial"):
        citation = f"Page {best.page}, chunk {best.chunk_index}"
        if best.section and best.section != "general":
            citation += f" ({best.section})"

    explanation, excerpt = _build_explanation(
        req_text=req_text,
        status=status,
        relevance=relevance,
        best=best,
        concepts=concepts,
    )

    return RequirementMatch(
        requirement=req_text,
        category=category,
        priority=priority,
        status=status,
        resume_citation=citation,
        resume_excerpt=excerpt,
        explanation=explanation,
        score=round(relevance * 100, 1),
    )
