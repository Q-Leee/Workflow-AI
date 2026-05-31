"""
Skill/concept ontology for semantic query expansion (not keyword-only matching).

Maps JD surface terms to related tools, platforms, and practices so retrieval
can find equivalent experience (e.g. GCP ↔ Vertex AI / BigQuery).
"""
from __future__ import annotations

import re
from functools import lru_cache

_KEYWORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}")

# concept_id -> aliases (lowercase); first alias is the canonical label
_ONTOLOGY: dict[str, list[str]] = {
    "gcp": [
        "gcp",
        "google cloud",
        "google cloud platform",
        "vertex ai",
        "bigquery",
        "cloud storage",
        "cloud run",
        "cloud sql",
        "iam",
        "gke",
        "kubernetes engine",
    ],
    "aws": [
        "aws",
        "amazon web services",
        "s3",
        "lambda",
        "ec2",
        "rds",
        "cloudformation",
    ],
    "azure": ["azure", "microsoft azure", "blob storage", "azure functions"],
    "rag": [
        "rag",
        "retrieval augmented",
        "retrieval-augmented",
        "vector search",
        "semantic search",
        "embedding retrieval",
        "embeddings",
        "chroma",
        "hybrid search",
        "bm25",
        "document qa",
        "question answering",
    ],
    "llm": [
        "llm",
        "large language model",
        "generative ai",
        "genai",
        "ollama",
        "prompt",
        "chat model",
    ],
    "mcp": [
        "mcp",
        "model context protocol",
        "json-rpc tools",
        "agent tools",
    ],
    "react": ["react", "react.js", "reactjs", "jsx", "frontend react"],
    "node": ["node.js", "nodejs", "node"],
    "python": ["python", "fastapi", "django", "flask", "pytest"],
    "java": [
        "java",
        "spring boot",
        "spring framework",
        "j2ee",
        "junit",
        "mockito",
    ],
    "sql": [
        "sql",
        "mysql",
        "postgresql",
        "postgres",
        "sqlite",
        "turso",
        "libsql",
        "mongodb",
        "nosql",
        "schema design",
    ],
    "snowflake": ["snowflake", "data warehouse", "snowsql"],
    "dbt": ["dbt", "dbt cloud", "data build tool"],
    "ml_framework": [
        "pytorch",
        "tensorflow",
        "opencv",
        "scikit-learn",
        "machine learning",
        "deep learning",
    ],
    "cv": [
        "computer vision",
        "image processing",
        "video analytics",
        "opencv",
    ],
    "nlp": [
        "nlp",
        "natural language",
        "speech recognition",
        "text processing",
        "language model",
    ],
    "playwright": ["playwright", "browser automation", "e2e testing"],
    "langchain": ["langchain", "langgraph", "crewai", "agentic"],
    "api": [
        "rest",
        "restful",
        "api",
        "http",
        "graphql",
        "openapi",
        "swagger",
        "fastapi",
    ],
    "cicd": [
        "ci/cd",
        "cicd",
        "github actions",
        "jenkins",
        "pipeline",
        "deployment automation",
    ],
    "fullstack": [
        "full stack",
        "full-stack",
        "frontend and backend",
        "end to end",
    ],
    "embedded_c": [
        "embedded c",
        "c programming",
        "firmware",
        "bare-metal",
        "bare metal",
        "microcontroller",
        "embedded linux",
        "memory management",
        "pointers",
        "data structures",
        "hardware bring-up",
        "bring-up",
    ],
    "embedded_test": [
        "unit test",
        "unit testing",
        "test plan",
        "validation",
    ],
    "low_code_automation": [
        "power automate",
        "microsoft power automate",
        "copilot",
        "copilot studio",
        "low-code",
        "no-code",
        "workflow automation",
        "power platform",
        "claude",
    ],
    "agentic_ai": [
        "agentic ai",
        "ai agents",
        "autonomous agents",
        "agentic systems",
        "multi-step agents",
    ],
    "web_stack": [
        "typescript",
        "php",
        "symfony",
        "aurelia",
        "react",
        "javascript",
    ],
    "ai_coding": [
        "ai coding tools",
        "cursor",
        "copilot",
        "github copilot",
        "ai-assisted development",
    ],
}

# JD phrase patterns -> concept ids (checked in order)
_REQ_TRIGGERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(gcp|google cloud|vertex ai)\b", re.I), "gcp"),
    (re.compile(r"\b(aws|amazon web)\b", re.I), "aws"),
    (re.compile(r"\bazure\b", re.I), "azure"),
    (re.compile(r"\brag\b|retrieval[- ]?augmented|vector search|semantic search", re.I), "rag"),
    (re.compile(r"\b(llm|genai|generative ai)\b", re.I), "llm"),
    (re.compile(r"\bmcp\b|model context protocol", re.I), "mcp"),
    (re.compile(r"\breact\b(?!\s*native)", re.I), "react"),
    (re.compile(r"\bnode\.?js\b", re.I), "node"),
    (re.compile(r"\bpython\b", re.I), "python"),
    (re.compile(r"\bjava\b(?!\s*script)", re.I), "java"),
    (re.compile(r"\b(sql|mysql|postgres|mongodb|database)\b", re.I), "sql"),
    (re.compile(r"\bsnowflake\b", re.I), "snowflake"),
    (re.compile(r"\bdbt\b", re.I), "dbt"),
    (re.compile(r"\b(pytorch|tensorflow|opencv|ai/ml)\b", re.I), "ml_framework"),
    (re.compile(r"\b(computer vision|image processing|video analytics)\b", re.I), "cv"),
    (re.compile(r"\b(nlp|natural language|pattern recognition)\b", re.I), "nlp"),
    (re.compile(r"\bdata mining\b", re.I), "ml_framework"),
    (re.compile(r"\bplaywright\b", re.I), "playwright"),
    (re.compile(r"\b(langchain|langgraph|crewai)\b", re.I), "langchain"),
    (re.compile(r"\b(restful?|web\s*api)\b", re.I), "api"),
    (re.compile(r"\bci/?cd\b", re.I), "cicd"),
    (re.compile(r"\bfull[- ]?stack\b", re.I), "fullstack"),
    (re.compile(r"\b(embedded\s+c|c programming|firmware|bare[- ]?metal|microcontroller)\b", re.I), "embedded_c"),
    (re.compile(r"\b(embedded\s+linux)\b", re.I), "embedded_c"),
    (re.compile(r"\b(unit tests?|test plans?)\b", re.I), "embedded_test"),
    (re.compile(r"\b(schematics?|electronic schematic)\b", re.I), "embedded_c"),
    (re.compile(r"\b(power automate|copilot(?:\s+studio)?)\b", re.I), "low_code_automation"),
    (re.compile(r"\blow[- ]?code|no[- ]?code\b", re.I), "low_code_automation"),
    (re.compile(r"\bagentic\b", re.I), "agentic_ai"),
    (re.compile(r"\b(typescript|php|symfony|aurelia)\b", re.I), "web_stack"),
    (re.compile(r"\bai coding tools?\b", re.I), "ai_coding"),
]

# Natural-language search templates per concept (retrieval queries)
_QUERY_TEMPLATES: dict[str, list[str]] = {
    "gcp": [
        "Google Cloud Platform GCP Vertex AI BigQuery Cloud Run deployment",
        "cloud infrastructure IAM managed services analytics",
    ],
    "aws": ["AWS cloud S3 Lambda deployment infrastructure"],
    "azure": ["Microsoft Azure cloud deployment hosting"],
    "rag": [
        "RAG retrieval augmented generation vector embeddings semantic search Chroma hybrid",
        "document question answering LLM embeddings BM25",
    ],
    "llm": ["LLM generative AI Ollama prompts embeddings NLP"],
    "mcp": ["Model Context Protocol MCP tools JSON-RPC agents"],
    "react": ["React TypeScript frontend UI components SPA"],
    "node": ["Node.js JavaScript backend server API"],
    "python": ["Python FastAPI backend programming data pipelines"],
    "java": ["Java Spring Boot JUnit enterprise backend"],
    "sql": ["SQL database MySQL PostgreSQL MongoDB schema persistence"],
    "snowflake": ["Snowflake data warehouse analytics SQL cloud"],
    "dbt": ["dbt transformations data modelling Git analytics engineering"],
    "ml_framework": ["PyTorch TensorFlow OpenCV machine learning model training"],
    "cv": ["computer vision image processing video analytics OpenCV"],
    "nlp": ["NLP natural language processing speech recognition text analytics"],
    "playwright": ["Playwright browser automation E2E web testing"],
    "langchain": ["LangChain LangGraph agentic AI orchestration"],
    "api": ["REST API HTTP FastAPI endpoints OpenAPI integration"],
    "cicd": ["CI/CD GitHub Actions deployment pipeline Agile"],
    "fullstack": ["full stack frontend backend production application"],
    "embedded_c": [
        "embedded C firmware bare-metal microcontroller embedded Linux pointers",
        "firmware development memory management data structures hardware",
    ],
    "embedded_test": ["unit testing test plans firmware validation embedded testing"],
    "low_code_automation": [
        "Microsoft Power Automate Copilot Studio low-code workflow automation integrations",
    ],
    "agentic_ai": [
        "agentic AI autonomous agents multi-step LLM agent evaluation traces",
    ],
    "web_stack": ["TypeScript PHP Symfony Aurelia React frontend web development"],
    "ai_coding": [
        "AI coding tools Cursor Copilot AI-assisted development everyday workflow",
    ],
}


@lru_cache(maxsize=256)
def concepts_for_requirement(req_text: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    lower = req_text.lower()
    for pat, cid in _REQ_TRIGGERS:
        if pat.search(req_text) and cid not in seen:
            seen.add(cid)
            found.append(cid)
    # Token overlap with ontology aliases
    for cid, aliases in _ONTOLOGY.items():
        if cid in seen:
            continue
        for alias in aliases:
            if len(alias) >= 4 and alias in lower:
                seen.add(cid)
                found.append(cid)
                break
    return tuple(found)


def expand_search_queries(req_text: str, *, category: str = "") -> list[str]:
    """Build retrieval queries: requirement text + ontology-expanded variants."""
    queries = [req_text.strip()]
    cat = (category or "").lower()
    if cat == "education":
        queries.append(
            "degree bachelor master education qualification computer science "
            "software engineering artificial intelligence"
        )
    elif cat == "experience":
        queries.append(
            "professional experience employment production delivered shipped "
            "responsibilities achievements"
        )

    for cid in concepts_for_requirement(req_text):
        for template in _QUERY_TEMPLATES.get(cid, []):
            queries.append(template)

    # Domain-style expansion from key tokens
    tokens = [t for t in _KEYWORD_RE.findall(req_text) if len(t) > 3]
    if len(tokens) >= 2:
        queries.append(" ".join(tokens[:8]) + " skills tools experience project")

    return list(dict.fromkeys(q for q in queries if q and len(q) >= 8))


def _alias_hits(text: str, concept_id: str) -> int:
    lower = text.lower()
    aliases = _ONTOLOGY.get(concept_id, [])
    hits = 0
    for alias in aliases:
        if len(alias) < 3:
            continue
        if " " in alias:
            if alias in lower:
                hits += 1
        elif re.search(rf"\b{re.escape(alias)}\b", lower):
            hits += 1
    return hits


def ontology_overlap_score(req_text: str, corpus_text: str) -> float:
    """
    0–1 score: how well resume text covers JD concepts via ontology (not exact string match).
    """
    concepts = concepts_for_requirement(req_text)
    if not concepts:
        return 0.0
    corpus = corpus_text.lower()
    scores: list[float] = []
    for cid in concepts:
        aliases = _ONTOLOGY.get(cid, [])
        if not aliases:
            continue
        hits = _alias_hits(corpus, cid)
        # Require at least one alias hit; scale by coverage
        if hits == 0:
            scores.append(0.0)
        else:
            scores.append(min(1.0, 0.35 + 0.15 * hits))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def resume_satisfies_concept(req_text: str, corpus_text: str, *, min_score: float = 0.35) -> bool:
    return ontology_overlap_score(req_text, corpus_text) >= min_score


def concept_labels(req_text: str) -> list[str]:
    return [c.upper().replace("_", " ") for c in concepts_for_requirement(req_text)]
