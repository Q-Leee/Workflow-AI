import logging
import re
from pathlib import Path

from app.config import settings
from app.models.schemas import SourceChunk
from app.services import auth_service, embeddings, hybrid_search, llm, reranker, vector_store

LOW_CONFIDENCE_ANSWER = (
    "I could not find enough relevant content in the document for this question. "
    "Try rephrasing, naming a section or page, or re-uploading a text-based PDF, Word, or Excel file."
)
from app.services.chunking import chunk_pdf_pages
from app.services.document_extract import SUPPORTED_EXTENSIONS, extract_pages_from_file
from app.services.pdf_extract import extract_pages_pdf

logger = logging.getLogger(__name__)

_SKILL_QUESTION_RE = re.compile(
    r"\bskills?\b|what\s+skill|tech\s*stack|programming\s+language|tools?\s+(?:does|do|he|she|they)",
    re.IGNORECASE,
)
_TECH_NAME_RE = re.compile(
    r"\b(?:Python|Java(?:Script)?|TypeScript|React(?:\s+Native)?|Node\.?js|Express\.?js|"
    r"FastAPI|Django|Flask|HTML5?|CSS3?|SQL|MySQL|PostgreSQL|MongoDB|Supabase|"
    r"PyTorch|TensorFlow|OpenCV|Ollama|Chroma(?:DB)?|Git|Docker|Kubernetes|AWS|Azure|GCP|"
    r"SAP|REST|GraphQL|Vite|BM25|LLM|RAG|Power\s+Automate|Google\s+Sheets|OAuth|"
    r"C(?:\+\+|#| programming)?|NumPy|Pandas|scikit-learn)\b",
    re.IGNORECASE,
)
_LIST_QUERY_RE = re.compile(
    r"\b(all|every|each|list|enumerate|complete|full)\b.*\b(experience|job|role|skill|position|employer)",
    re.IGNORECASE,
)
_LIST_GENERAL_RE = re.compile(
    r"\b(all|every|each|list|enumerate|complete|full|how many|summarize all)\b",
    re.IGNORECASE,
)
_SECTION_QUERY_RE = re.compile(
    r"\b(section|chapter|article|clause|paragraph|appendix|schedule)\b",
    re.IGNORECASE,
)


def _extract_urls(text: str) -> list[str]:
    # Regex to capture fully qualified http/https URLs.
    url_pattern = re.compile(r'https?://[^\s]+')
    found_urls = url_pattern.findall(text)
    
    cleaned_urls = []
    for u in found_urls:
        # Strip trailing punctuation marks that might be part of the sentence structure but not the URL
        cleaned = u.strip(".,;:?!()[]{}'\"")
        if cleaned:
            cleaned_urls.append(cleaned)
    return list(dict.fromkeys(cleaned_urls)) # Deduplicate


def _is_list_style_question(question: str) -> bool:
    q = question.strip()
    return bool(_LIST_QUERY_RE.search(q) or _LIST_GENERAL_RE.search(q))


def _effective_top_k(question: str, top_k: int) -> int:
    if _is_list_style_question(question):
        return min(30, max(top_k, settings.query_top_k_list))
    if _is_skill_question(question):
        return min(30, max(top_k, 12))
    return min(30, max(top_k, settings.query_top_k_default))


def _search_queries(question: str) -> list[str]:
    q = question.strip()
    queries = [q]
    if _LIST_QUERY_RE.search(q) or re.search(r"\bwork experience\b", q, re.I):
        queries.append(
            "employment history job positions companies dates responsibilities achievements"
        )
    if _LIST_GENERAL_RE.search(q):
        queries.append("complete list all items every entry details requirements")
    if _SECTION_QUERY_RE.search(q):
        queries.append("section heading title contents summary")
    if _SKILL_QUESTION_RE.search(q):
        queries.append(
            "technical skills programming languages tools frameworks libraries databases cloud DevOps"
        )
    if re.search(r"\b(policy|procedure|requirement|must|shall)\b", q, re.I):
        queries.append("policy rules obligations requirements compliance")
    return list(dict.fromkeys(queries))


def _is_skill_question(question: str) -> bool:
    return bool(_SKILL_QUESTION_RE.search(question))


def _sources_have_skill_content(sources: list[SourceChunk]) -> bool:
    blob = " ".join(s.text for s in sources[:8])
    return bool(_TECH_NAME_RE.search(blob)) or bool(
        re.search(r"\b(?:skills?|frameworks?|languages?|technologies)\b", blob, re.I)
    )


def _answer_skills_from_sources(sources: list[SourceChunk]) -> str | None:
    """Deterministic skill list from retrieved chunks when LLM is unavailable or too strict."""
    found: dict[str, int] = {}
    for idx, s in enumerate(sources, start=1):
        for m in _TECH_NAME_RE.finditer(s.text):
            name = m.group(0)
            key = name.lower()
            if key not in found:
                found[key] = idx
        for line in s.text.splitlines():
            if not re.search(r"\b(?:skills?|frameworks?|languages?|technologies)\b", line, re.I):
                continue
            tail = line.split(":", 1)[-1] if ":" in line else line
            for part in re.split(r"[,;|•·]", tail):
                token = part.strip(" \t\u2022\u00b7&")
                if len(token) >= 2 and not token.lower().startswith("and "):
                    key = token.lower()
                    if key not in found:
                        found[key] = idx

    if not found:
        return None

    items = sorted(found.items(), key=lambda x: x[1])
    lines = [
        "Technical skills and tools mentioned in the resume (from uploaded document):",
        "",
    ]
    for name, cite in items[:40]:
        display = name if name[0].isupper() else name.title()
        lines.append(f"- {display} [{cite}]")
    lines.append("")
    lines.append(
        "Tip: For a fuller narrative answer, ensure Ollama is running "
        f"with model `{settings.ollama_chat_model}`."
    )
    return "\n".join(lines)


def best_source_relevance(sources: list[SourceChunk]) -> float:
    best = 0.0
    for s in sources:
        rel = reranker.relevance_0_1(
            rerank_score=s.rerank_score,
            vector_score=s.vector_score,
        )
        best = max(best, rel)
    return best


def _ingest_chunks(
    *,
    user_id: str,
    document_id: str,
    filename: str,
    doc_type: str,
    chunks: list,
) -> int:
    if not chunks:
        vector_store.get_collection()
        return 0
    texts = [c.text for c in chunks]
    pages_out = [c.page_start for c in chunks]
    indices = [c.chunk_index for c in chunks]
    sections = [getattr(c, "section", "general") for c in chunks]
    embs = embeddings.embed_texts(texts)
    vector_store.upsert_chunks(
        user_id=user_id,
        document_id=document_id,
        filename=filename,
        doc_type=doc_type,
        texts=texts,
        pages=pages_out,
        chunk_indices=indices,
        sections=sections,
        embeddings=embs,
    )
    return len(chunks)


def ingest_document_file(
    *,
    file_path: Path,
    original_filename: str,
    user_id: str,
    doc_type: str = "pdf",
) -> tuple[str, int]:
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    doc_id = vector_store.new_document_id()
    if ext == ".pdf":
        pages = extract_pages_pdf(file_path)
    else:
        pages = extract_pages_from_file(file_path, original_filename)
    chunks = chunk_pdf_pages(pages)
    n = _ingest_chunks(
        user_id=user_id,
        document_id=doc_id,
        filename=original_filename,
        doc_type=doc_type,
        chunks=chunks,
    )
    return doc_id, n


def ingest_pdf_file(
    *,
    file_path: Path,
    original_filename: str,
    user_id: str,
    doc_type: str = "pdf",
) -> tuple[str, int]:
    return ingest_document_file(
        file_path=file_path,
        original_filename=original_filename,
        user_id=user_id,
        doc_type=doc_type,
    )


def ingest_plain_text(
    *,
    text: str,
    original_filename: str,
    user_id: str,
    doc_type: str,
) -> tuple[str, int]:
    from app.services.chunking import chunk_page_text

    doc_id = vector_store.new_document_id()
    t = text.replace("\r\n", "\n")
    if not t.strip():
        return doc_id, 0
    chunks, _ = chunk_page_text(1, t, 0)
    n = _ingest_chunks(
        user_id=user_id,
        document_id=doc_id,
        filename=original_filename,
        doc_type=doc_type,
        chunks=chunks,
    )
    return doc_id, n


def _expand_page_context(
    hits: list[SourceChunk],
    *,
    user_id: str,
    document_id: str | None,
) -> list[SourceChunk]:
    if not settings.page_context_expand or not hits:
        return hits

    pages_seen: set[tuple[str, int]] = set()
    expanded: list[SourceChunk] = []
    for h in hits:
        key = (h.document_id, h.page)
        if key in pages_seen:
            expanded.append(h)
            continue
        pages_seen.add(key)
        where = vector_store.build_where_filter(
            user_id=user_id,
            document_id=document_id or h.document_id,
            filename=None,
            doc_type=None,
            page_min=h.page,
            page_max=h.page,
        )
        ids, docs, metas = vector_store.fetch_chunks(where, limit=50)
        if len(docs) <= 1:
            expanded.append(h)
            continue
        ordered = sorted(
            zip(ids, docs, metas),
            key=lambda x: int((x[2] or {}).get("chunk_index", 0)),
        )
        merged = "\n".join(d for d in docs if d)
        meta0 = metas[0] or {}
        expanded.append(
            SourceChunk(
                chunk_id=ids[0],
                document_id=str(meta0.get("document_id", h.document_id)),
                filename=str(meta0.get("filename", h.filename)),
                page=h.page,
                chunk_index=int(meta0.get("chunk_index", 0)),
                section=h.section,
                vector_score=h.vector_score,
                rerank_score=h.rerank_score,
                text=merged[: settings.page_context_max_chars],
            )
        )
    return expanded


def _retrieve_one(
    *,
    user_id: str,
    question: str,
    document_id: str | None,
    filename: str | None,
    doc_type: str | None,
    page_min: int | None,
    page_max: int | None,
    top_k: int,
) -> list[tuple[str, str, dict, float]]:
    """Returns list of (id, doc, meta, score) before rerank."""
    where = vector_store.build_where_filter(
        user_id=user_id,
        document_id=document_id,
        filename=filename,
        doc_type=doc_type,
        page_min=page_min,
        page_max=page_max,
    )
    n_fetch = max(top_k * settings.retrieval_multiplier, top_k + 5)

    q_emb = embeddings.embed_query(question)
    v_ids, v_docs, v_metas, v_scores = vector_store.query_similar(
        q_emb, top_k=n_fetch, where=where
    )

    if settings.hybrid_search_enabled and document_id:
        all_ids, all_docs, all_metas = vector_store.fetch_chunks(
            where, limit=settings.hybrid_pool_limit
        )
        pool_ids = list(v_ids)
        pool_docs = list(v_docs)
        pool_metas = list(v_metas)
        pool_scores = list(v_scores)
        seen = set(pool_ids)
        for i, cid in enumerate(all_ids):
            if cid not in seen:
                seen.add(cid)
                pool_ids.append(cid)
                pool_docs.append(all_docs[i])
                pool_metas.append(all_metas[i])
                pool_scores.append(0.0)
        if len(pool_ids) > len(v_ids):
            order = hybrid_search.hybrid_rank(
                question,
                chunk_ids=pool_ids,
                documents=pool_docs,
                vector_scores=pool_scores,
                top_k=n_fetch,
            )
            return [
                (pool_ids[i], pool_docs[i], pool_metas[i], pool_scores[i])
                for i in order
            ]

    if settings.hybrid_search_enabled and v_ids:
        order = hybrid_search.hybrid_rank(
            question,
            chunk_ids=v_ids,
            documents=v_docs,
            vector_scores=v_scores,
            top_k=n_fetch,
        )
        return [(v_ids[i], v_docs[i], v_metas[i], v_scores[i]) for i in order]

    return list(zip(v_ids, v_docs, v_metas, v_scores))


def retrieve_and_rerank(
    *,
    user_id: str,
    question: str,
    document_id: str | None,
    filename: str | None,
    doc_type: str | None,
    page_min: int | None,
    page_max: int | None,
    top_k: int,
    do_rerank: bool,
    expand_pages: bool | None = None,
) -> list[SourceChunk]:
    # 1. Detect URLs inside the question and run JIT web crawler pipeline
    urls = _extract_urls(question)
    crawled_doc_ids = []
    
    for url in urls:
        # Check ChromaDB cache first by looking for filename == url and doc_type == "web"
        try:
            where_url = vector_store.build_where_filter(
                user_id=user_id,
                document_id=None,
                filename=url,
                doc_type="web",
                page_min=None,
                page_max=None
            )
            c_ids, _, metas = vector_store.fetch_chunks(where_url, limit=1)
            if c_ids and metas:
                existing_doc_id = metas[0].get("document_id")
                if existing_doc_id:
                    crawled_doc_ids.append(existing_doc_id)
                    logger.info(f"RAG Cache Hit for Web URL: {url} (doc_id: {existing_doc_id})")
                    continue
        except Exception as e:
            logger.warning(f"Error checking web cache for URL {url}: {e}")
            
        # Cache Miss: Scrape and Ingest
        try:
            logger.info(f"RAG Cache Miss. Initiating dynamic scrape for URL: {url}")
            from app.services.web_crawler import crawl_url_to_markdown
            markdown_content = crawl_url_to_markdown(url)
            
            if markdown_content:
                # Ingest clean markdown plain text dynamically into ChromaDB
                crawled_doc_id, n_chunks = ingest_plain_text(
                    text=markdown_content,
                    original_filename=url,
                    user_id=user_id,
                    doc_type="web"
                )
                
                if n_chunks > 0:
                    crawled_doc_ids.append(crawled_doc_id)
                    # Register document into User's Document Database
                    auth_service.register_document(
                        user_id=user_id,
                        document_id=crawled_doc_id,
                        filename=url,
                        doc_type="web",
                        chunks=n_chunks
                    )
                    logger.info(f"JIT Ingestion Success: Injected {n_chunks} chunks for web resource {url}")
        except Exception as e:
            logger.exception(f"JIT Web Ingestion failed for url: {url}")
            
    # 2. Setup Multi-Source retrieval target document IDs
    target_doc_ids = []
    if document_id:
        target_doc_ids.append(document_id)
        
    for cid in crawled_doc_ids:
        if cid not in target_doc_ids:
            target_doc_ids.append(cid)
            
    if not target_doc_ids:
        target_doc_ids.append(None)
        
    fetch_k = _effective_top_k(question, top_k)
    merged_pool: dict[str, tuple[str, str, dict, float]] = {}

    # Run retrieval loops over all target document resources
    for doc_id_target in target_doc_ids:
        for q in _search_queries(question):
            try:
                for cid, doc, meta, score in _retrieve_one(
                    user_id=user_id,
                    question=q,
                    document_id=doc_id_target,
                    filename=filename if not doc_id_target else None,
                    doc_type=doc_type if not doc_id_target else None,
                    page_min=page_min if not doc_id_target else None,
                    page_max=page_max if not doc_id_target else None,
                    top_k=fetch_k * 2,
                ):
                    if cid not in merged_pool or score > merged_pool[cid][3]:
                        merged_pool[cid] = (cid, doc, meta, score)
            except Exception:
                logger.exception("Retrieval failed for query=%s on doc_id=%s", q, doc_id_target)

    if not merged_pool:
        return []

    items = sorted(merged_pool.values(), key=lambda x: x[3], reverse=True)
    ids = [x[0] for x in items]
    docs = [x[1] for x in items]
    metas = [x[2] for x in items]
    vec_scores = [x[3] for x in items]

    rerank_scores: list[float | None] = [None] * len(docs)
    if do_rerank and docs:
        rerank_scores = reranker.rerank_pairs(question, docs)

    ranked: list[tuple[float, int]] = []
    for i in range(len(ids)):
        primary = rerank_scores[i] if rerank_scores[i] is not None else vec_scores[i]
        ranked.append((primary, i))
    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[:fetch_k]

    results: list[SourceChunk] = []
    for _, i in ranked:
        meta = metas[i] or {}
        results.append(
            SourceChunk(
                chunk_id=ids[i],
                document_id=str(meta.get("document_id", "")),
                filename=str(meta.get("filename", "")),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                section=str(meta.get("section") or "general"),
                vector_score=float(vec_scores[i]) if i < len(vec_scores) else None,
                rerank_score=float(rerank_scores[i]) if rerank_scores[i] is not None else None,
                text=docs[i],
            )
        )

    if expand_pages if expand_pages is not None else settings.page_context_expand:
        return _expand_page_context(
            results,
            user_id=user_id,
            document_id=document_id,
        )
    return results


def build_context_blocks(sources: list[SourceChunk]) -> str:
    lines: list[str] = []
    for n, s in enumerate(sources, start=1):
        lines.append(f"[{n}] ({s.filename}, page {s.page}): {s.text}")
    return "\n\n".join(lines)


def _extract_facts_ollama(question: str, context: str) -> str | None:
    return llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Extract only facts from the sources that help answer the question. "
                    "Use bullet points. Each bullet must end with a citation like [1] or [2]. "
                    "If nothing is relevant, respond with exactly: NO_RELEVANT_FACTS"
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion:\n{question}",
            },
        ]
    )


def _answer_from_facts_ollama(question: str, facts: str) -> str | None:
    return llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise document assistant. Answer ONLY using the extracted facts. "
                    "Keep citations [1], [2] on every claim. "
                    "If facts are insufficient, say you don't know. "
                    "For list questions, include every distinct item from the facts."
                ),
            },
            {
                "role": "user",
                "content": f"Extracted facts:\n{facts}\n\nQuestion:\n{question}",
            },
        ]
    )


def _is_refusal_answer(text: str | None) -> bool:
    if not text:
        return True
    lower = text.lower()
    return (
        "could not find enough relevant content" in lower
        or lower.strip() == "no_relevant_facts"
        or "i don't know" in lower
        or "i do not know" in lower
    )


def _should_try_llm(sources: list[SourceChunk], question: str) -> bool:
    rel = best_source_relevance(sources)
    if rel >= settings.answer_min_relevance:
        return True
    if len(sources) >= 3 and _is_skill_question(question) and _sources_have_skill_content(sources):
        return True
    return len(sources) >= 5 and rel >= 0.12


def generate_answer_ollama(question: str, sources: list[SourceChunk]) -> str | None:
    """Generate an answer or None if confidence is too low."""
    if not sources:
        return None

    skill_q = _is_skill_question(question)
    fallback = (
        _answer_skills_from_sources(sources)
        if skill_q and _sources_have_skill_content(sources)
        else None
    )

    # Skill lists: never block on relevance gate when sources already name technologies.
    if skill_q and fallback:
        if not _should_try_llm(sources, question):
            return fallback

        context = build_context_blocks(sources)
        answer = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a precise document assistant. Answer ONLY using the provided sources. "
                        "Use inline citations [1], [2] for every claim. "
                        "For skill questions, list every programming language, framework, tool, and platform named. "
                        "Do not say content is missing when skills are listed in the sources."
                    ),
                },
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion:\n{question}"},
            ]
        )
        if answer and not _is_refusal_answer(answer):
            return answer
        return fallback

    if not _should_try_llm(sources, question):
        return fallback or LOW_CONFIDENCE_ANSWER

    context = build_context_blocks(sources)
    use_two_pass = settings.answer_two_pass_enabled and not skill_q

    if use_two_pass:
        facts = _extract_facts_ollama(question, context)
        if facts and facts.strip() != "NO_RELEVANT_FACTS":
            answer = _answer_from_facts_ollama(question, facts)
            if answer and not _is_refusal_answer(answer):
                return answer

    answer = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise document assistant. Answer ONLY using the provided sources. "
                    "If information is missing, say you don't know. "
                    "Use inline citations [1], [2] for every claim. "
                    "Quote numbers, dates, and clause IDs exactly as written. "
                    "When asked to list items, include EVERY distinct item across all sources. "
                    "For skill questions, list every programming language, framework, tool, and platform named."
                ),
            },
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion:\n{question}"},
        ]
    )
    if answer and not _is_refusal_answer(answer):
        return answer
    return fallback or (None if not answer else answer)
