"""
Generalized requirement knowledge — field/skill equivalence and junk filtering.

Like a small knowledge graph (not per-JD patches): CS ≈ Software Engineering ≈ AI,
Python on resume satisfies "Python or JavaScript", etc.
"""
from __future__ import annotations

import re

# Study / discipline clusters (any alias in JD + any alias in resume → match)
STEM_DEGREE_ALIASES: frozenset[str] = frozenset(
    {
        "computer science",
        "software engineering",
        "artificial intelligence",
        "machine learning",
        "data science",
        "information technology",
        "information systems",
        "mathematics",
        "computer engineering",
        "informatics",
        "cyber security",
        "cybersecurity",
        "it",
        "computing",
        "engineering",
        "related discipline",
        "related field",
    }
)

_DEGREE_REQ = re.compile(
    r"\b(bachelor|master|phd|doctorate|degree|graduate|diploma|studying|completed|"
    r"pursuing|computer science|software engineering|artificial intelligence|"
    r"data science|information technology|related)\b",
    re.I,
)

# Non-scorable: benefits, culture marketing, incomplete section intros
_JUNK_REQUIREMENT = re.compile(
    r"(work[- ]life balance|competitive compensation|comprehensive benefits|"
    r"accelerated professional development|collaborative.{0,30}environment|"
    r"varied and impactful projects|who are you joining|deloitte tech fast|"
    r"fast starters|hashtag#|#tranzformd|if you.?re ready to make a difference|"
    r"enjoy a competitive salary|design a schedule that works for you|"
    r"you might be studying|we.?re less interested in|most importantly, you.?re|"
    r"natural(?:ly)? curious and excited|wants to work directly with clients|"
    r"privacy policy|applicant privacy|application limit|apply early|"
    r"rolling basis|equal opportunity|who are we\?|about the job\s*$|"
    r"team introduction|building a world where people)",
    re.I,
)

_INCOMPLETE_REQ = re.compile(
    r"^(you might be studying|we.?d love to see|what we.?re looking for|"
    r"minimum qualifications|preferred qualifications|qualifications)\s*[:\-]?\s*$",
    re.I,
)

# JD fields mentioned in a list → any STEM degree on resume counts
_DEGREE_LIST = re.compile(
    r"\b(computer science|software engineering|artificial intelligence|"
    r"data science|information technology|mathematics|related disciplines?)\b",
    re.I,
)

_YEARS_REQ = re.compile(r"\b(\d+)\s*[-–—]?\s*(\d+)?\s*years?\s+(of\s+)?experience\b", re.I)


def is_junk_requirement(text: str) -> bool:
    t = text.strip()
    if len(t) < 12:
        return True
    if _INCOMPLETE_REQ.match(t):
        return True
    if _JUNK_REQUIREMENT.search(t):
        return True
    # Marketing sentence without a concrete skill/degree ask
    if len(t) > 80 and not _DEGREE_REQ.search(t):
        if not re.search(
            r"\b(python|javascript|java|sql|aws|azure|gcp|llm|machine learning|"
            r"tensorflow|pytorch|api|git|cloud|nlp|computer vision)\b",
            t,
            re.I,
        ):
            if re.search(r"\b(we understand|you.?ll join|committed to helping|values)\b", t, re.I):
                return True
    return False


def _resume_has_stem_degree(resume_lower: str) -> bool:
    if not re.search(r"\b(bachelor|master|b\.?s\.?c|b\.?eng|degree|software engineering)\b", resume_lower):
        return False
    if any(alias in resume_lower for alias in STEM_DEGREE_ALIASES):
        return True
    if re.search(r"\b(torrens|university|college)\b", resume_lower) and re.search(
        r"\b(engineering|artificial intelligence|computer)\b", resume_lower
    ):
        return True
    return False


def education_match_score(req_text: str, resume_text: str) -> float:
    """
    0–1: does resume education satisfy JD degree / field requirement?
    CS JD line + Software Engineering (AI) degree → high score.
    """
    req = req_text.lower()
    resume = resume_text.lower()

    if not _DEGREE_REQ.search(req):
        return 0.0
    if not _resume_has_stem_degree(resume):
        return 0.0

    # PhD/Master-only lines when candidate has Bachelor
    if re.search(r"\b(phd|doctorate)\b", req) and not re.search(r"\b(phd|doctorate)\b", resume):
        if re.search(r"\b(bachelor|b\.s\.c|undergraduate)\b", resume):
            return 0.28
        return 0.0

    if re.search(r"\b(currently pursuing|completed within|pursuing your)\b.*\bbachelor", req):
        if re.search(r"\b(present|2024|2025|completed|bachelor|degree)\b", resume):
            return 0.58

    jd_fields = set(_DEGREE_LIST.findall(req))
    if jd_fields or re.search(r"\brelated\b", req):
        return 0.62

    if re.search(r"\b(bachelor|degree|graduate)\b", req):
        return 0.55

    return 0.45


def _req_signature(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower().strip())
    t = re.sub(r"[^\w\s+#.-]", "", t)
    return t[:100]


def _same_education_cluster(a: str, b: str) -> bool:
    a_lower = a.lower()
    b_lower = b.lower()
    if not (_DEGREE_REQ.search(a_lower) and _DEGREE_REQ.search(b_lower)):
        return False
    
    # If one specifies PhD/Master and the other Bachelor, they are different clusters
    a_high = bool(re.search(r"\b(phd|doctorate|master|ms\b|msc\b|postgraduate)\b", a_lower))
    b_high = bool(re.search(r"\b(phd|doctorate|master|ms\b|msc\b|postgraduate)\b", b_lower))
    if a_high != b_high:
        return False
    return True


_ML_CLUSTER = re.compile(
    r"\b(machine learning|pattern recognition|nlp|data mining|computer vision|"
    r"tensorflow|pytorch)\b",
    re.I,
)


def _same_ml_cluster(a: str, b: str) -> bool:
    a_lower = a.lower()
    b_lower = b.lower()
    if bool(re.search(r"\b(nlp|natural language)\b", a_lower)) and bool(re.search(r"\b(nlp|natural language)\b", b_lower)):
        return True
    if bool(re.search(r"\b(computer vision|opencv|image processing|video analytics)\b", a_lower)) and bool(re.search(r"\b(computer vision|opencv|image processing|video analytics)\b", b_lower)):
        return True
    if bool(re.search(r"\b(pytorch|tensorflow|keras|deep learning)\b", a_lower)) and bool(re.search(r"\b(pytorch|tensorflow|keras|deep learning)\b", b_lower)):
        return True
    if bool(re.search(r"\b(machine learning|data mining|pattern recognition)\b", a_lower)) and bool(re.search(r"\b(machine learning|data mining|pattern recognition)\b", b_lower)):
        return True
    return False


def merge_semantic_clusters(items: list[dict]) -> list[dict]:
    """One row per knowledge cluster (education, ML/NLP/CV) where possible."""
    if not items:
        return []

    kept: list[dict] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        category = item.get("category")
        merged = False
        for i, existing in enumerate(kept):
            ex = str(existing.get("text") or "")
            ex_category = existing.get("category")
            # NEVER merge requirements from different categories
            if category != ex_category:
                continue
            if not (_same_education_cluster(text, ex) or _same_ml_cluster(text, ex)):
                continue
            merged = True
            winner = existing
            if item.get("priority") == "required" and existing.get("priority") != "required":
                winner = item
            elif _same_ml_cluster(text, ex):
                winner = item if len(text) >= len(ex) else existing
            elif _same_education_cluster(text, ex):
                if item.get("priority") == "required":
                    winner = item
                else:
                    winner = item if len(text) <= len(ex) else existing
            kept[i] = winner
            break
        if not merged:
            kept.append(item)
    return kept


def dedupe_requirements(items: list[dict]) -> list[dict]:
    """Merge duplicate / overlapping requirements (e.g. three CS-degree lines)."""
    if not items:
        return []

    kept: list[dict] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if is_junk_requirement(text):
            continue
        sig = _req_signature(text)
        duplicate = False
        for i, existing in enumerate(kept):
            ex_text = str(existing.get("text") or "")
            ex_sig = _req_signature(ex_text)
            if sig == ex_sig or sig in ex_sig or ex_sig in sig:
                duplicate = True
                # If existing came from LLM but new one does not, do NOT overwrite it
                if existing.get("from_llm", False) and not item.get("from_llm", False):
                    break
                # If new one came from LLM but existing does not, ALWAYS overwrite it
                if item.get("from_llm", False) and not existing.get("from_llm", False):
                    kept[i] = item
                    break
                if len(text) > len(ex_text):
                    kept[i] = item
                break
            if _same_education_cluster(text, ex_text):
                duplicate = True
                if item.get("priority") == "required" and existing.get("priority") != "required":
                    kept[i] = item
                break
        if not duplicate:
            kept.append(item)
    return merge_semantic_clusters(kept)


def find_skill_evidence_chunk(
    req_text: str,
    resume_chunks: list,
) -> object | None:
    """Best chunk mentioning skills from requirement_knowledge floor rules."""
    if not resume_chunks:
        return None
    req = req_text.lower()
    patterns: list[re.Pattern] = []
    if re.search(r"\bpython\b", req, re.I):
        patterns.append(re.compile(r"\bpython\b", re.I))
    if re.search(r"\b(machine learning|nlp|computer vision|pytorch|tensorflow|data mining)\b", req, re.I):
        patterns.append(
            re.compile(
                r"\b(python|pytorch|tensorflow|opencv|nlp|machine learning|computer vision|"
                r"scikit|pandas|numpy)\b",
                re.I,
            )
        )
    if re.search(r"\b(computer vision|opencv)\b", req, re.I):
        patterns.append(re.compile(r"\b(opencv|computer vision|image processing)\b", re.I))
    if _DEGREE_REQ.search(req):
        patterns.append(re.compile(r"\b(bachelor|degree|engineering|university|torrens)\b", re.I))

    for pat in patterns:
        for ch in resume_chunks:
            if pat.search(ch.text):
                return ch
    return None


def corpus_skill_floor(req_text: str, resume_text: str) -> float:
    """
    General skill floors from resume corpus (not per-JD).
    Returns minimum relevance if resume clearly contains the skill family.
    """
    resume = resume_text.lower()
    req = req_text.lower()

    checks: list[tuple[re.Pattern, re.Pattern, float]] = [
        (re.compile(r"\bpython\b", re.I), re.compile(r"\bpython\b", re.I), 0.55),
        (re.compile(r"\b(javascript|typescript)\b", re.I), re.compile(r"\b(javascript|typescript)\b", re.I), 0.52),
        (re.compile(r"\bjava\b(?!script)", re.I), re.compile(r"\bjava\b", re.I), 0.48),
        (
            re.compile(r"\b(machine learning|pattern recognition|nlp|data mining|computer vision)\b", re.I),
            re.compile(
                r"\b(machine learning|pytorch|tensorflow|opencv|nlp|scikit|computer vision|"
                r"data mining|pandas|numpy)\b",
                re.I,
            ),
            0.55,
        ),
        (
            re.compile(r"\b(nlp|natural language)\b", re.I),
            re.compile(r"\b(nlp|speech recognition|language)\b", re.I),
            0.52,
        ),
        (
            re.compile(r"\b(computer vision|opencv)\b", re.I),
            re.compile(r"\b(computer vision|opencv|image processing)\b", re.I),
            0.55,
        ),
        (
            re.compile(r"\b(pytorch|tensorflow)\b", re.I),
            re.compile(r"\b(pytorch|tensorflow)\b", re.I),
            0.55,
        ),
        (
            re.compile(r"\b(generative ai|llm|llms)\b", re.I),
            re.compile(r"\b(llm|generative|ollama|rag|prompt|embeddings?)\b", re.I),
            0.55,
        ),
        (
            re.compile(r"\b(prompt engineering|model evaluation|ai testing)\b", re.I),
            re.compile(r"\b(prompt|evaluation|testing|llm|rag)\b", re.I),
            0.48,
        ),
        (
            re.compile(r"\b(data engineering|analytics)\b", re.I),
            re.compile(r"\b(pandas|numpy|sql|etl|analytics|data preprocessing|turso|mongodb)\b", re.I),
            0.45,
        ),
        (
            re.compile(r"\b(aws|azure|gcp|cloud platforms?)\b", re.I),
            re.compile(r"\b(aws|azure|gcp|google cloud|microsoft azure|amazon web)\b", re.I),
            0.38,
        ),
        (
            re.compile(r"\bdata mining|data visualization\b", re.I),
            re.compile(r"\b(pandas|numpy|data preprocessing|analytics|visualiz)\b", re.I),
            0.48,
        ),
        (
            re.compile(r"\bdata structures|algorithms\b", re.I),
            re.compile(r"\b(c programming|algorithm|data structure|database systems)\b", re.I),
            0.40,
        ),
        (
            re.compile(r"\b(database|sql|nosql)\b", re.I),
            re.compile(r"\b(database|sql|nosql|postgres|mysql|sqlite|mongodb|oracle)\b", re.I),
            0.48,
        ),
        (
            re.compile(r"\b(proficiency|programming language)\b", re.I),
            re.compile(r"\b(python|javascript|typescript|java)\b", re.I),
            0.52,
        ),
        (
            re.compile(r"\b(react|frontend|front[- ]end|web developer)\b", re.I),
            re.compile(r"\b(react|react native|frontend|front[- ]end)\b", re.I),
            0.55,
        ),
        (
            re.compile(r"\b(agile|scrum|ci/cd|cicd|project management|sprint)\b", re.I),
            re.compile(r"\b(agile|scrum|github actions|jenkins|gitlab|ci/cd|cicd)\b", re.I),
            0.48,
        ),
        (
            re.compile(r"\b(node\.?js|nest\.?js|express\.?js)\b", re.I),
            re.compile(r"\b(node\.?js|nest\.?js|express\.?js)\b", re.I),
            0.52,
        ),
        (
            re.compile(r"\b(java|spring\s*boot)\b", re.I),
            re.compile(r"\b(java|spring\s*boot)\b", re.I),
            0.48,
        ),
    ]
    floor = 0.0
    for req_pat, res_pat, score in checks:
        if req_pat.search(req) and res_pat.search(resume):
            floor = max(floor, score)

    # Cap multi-stack / multi-technology requirements if candidate only has one part of it
    if floor > 0:
        stacks = [
            (re.compile(r"\b(react|reactjs)\b", re.I), "React"),
            (re.compile(r"\b(next\.?js)\b", re.I), "Next.js"),
            (re.compile(r"\b(node\.?js|nest\.?js|express\.?js)\b", re.I), "Node.js"),
            (re.compile(r"\b(java|spring\s*boot)\b", re.I), "Java"),
            (re.compile(r"\b(python)\b", re.I), "Python"),
        ]
        req_stacks = [s for s in stacks if s[0].search(req)]
        if len(req_stacks) >= 2:
            missing = [s for s in req_stacks if not s[0].search(resume)]
            if missing:
                floor = min(floor, 0.38)

    return floor
