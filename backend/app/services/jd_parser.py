import json
import logging
import re

from app.config import settings
from app.db.database import get_conn
from app.services import jd_extract, llm, requirement_knowledge
from app.services.content_hash import hash_text, normalize_text

logger = logging.getLogger(__name__)

# Lines that are benefits/culture/marketing — not resume-matchable requirements
_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"https?://",
        r"www\.",
        r"linkedin\.com",
        r"visit our website",
        r"life page",
        r"awesome video",
        r"great place to work",
        r"certified great",
        r"our culture",
        r"moose magic",
        r"moose happy",
        r"breakfast bar",
        r"veggie garden",
        r"\bgym\b",
        r"yoga class",
        r"massage",
        r"flu vax",
        r"skin check",
        r"bring your pet",
        r"wellbeing day",
        r"school holiday program",
        r"ev charging",
        r"health insurance",
        r"employee assistance",
        r"flexible hybrid",
        r"wfh and flexi",
        r"social events",
        r"guest speakers",
        r"volunteer drive",
        r"thought leaders",
        r"present live in our",
        r"discount(ed)? private",
        r"free employee",
        r"daily breakfast",
        r"onsite fully serviced",
        r"circuit training",
        r"1:1 pt\b",
        r"click the.?apply",
        r"quote the above job reference",
        r"contact .+ on \d",
        r"@[\w.-]+\.(com|au)\b",
        r"paxus values diversity",
        r"indigenous australians",
        r"adjustment to the recruitment",
        r"alternate format",
        r"to be considered for the role",
        r"job reference number",
        r"\d+-month initial contract",
        r"high chance of extension",
        r"melbourne-based hybrid",
        r"opportunity to work on a large-scale",
    ]
]

_RECRUITMENT_ONLY = re.compile(
    r"(paxus|randstad|diversity and welcomes|equal employment opportunit|"
    r"embracing diversity|actively encourage applications|indigenous australians|"
    r"disability|adjustment to the recruitment|click the.?apply|quote the above|"
    r"contact .+ on \d{2}|yjain@|job reference number|to be considered|"
    r"applications from any background)",
    re.IGNORECASE,
)

_OFFER_LINE = re.compile(
    r"(what.?s on offer|on offer\b|hybrid working model|initial \d+-month contract|"
    r"opportunity to work on a large-scale|melbourne-based|collaborative team environment|"
    r"modern delivery practices\b(?!\s+with\s+(java|react|python)))",
    re.IGNORECASE,
)

_SECTION_HEADER_ONLY = re.compile(
    r"^(requirements?|qualifications?|key responsibilities|about the role|"
    r"skills?(\s+required)?|experience|education|must have|nice to have|"
    r"what you.?ll need|what you will learn|the role|about you)\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

_SKILLS_SECTION_START = re.compile(
    r"^(?:"
    r"skills?(?:\s+required|\s+&|\s+and)?\s*(?:experience)?|"
    r"key skills?(?:\s*(?:&|and)\s*experience)?|"
    r"required skills?|technical skills|technical stack|"
    r"experience required|must have|essential criteria"
    r")\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

_SUBSECTION_HEADER_ONLY = re.compile(
    r"^(ai connectivity|agentic ai|ai search|data engineering|web automation|"
    r"geo\b|required skills)\b",
    re.IGNORECASE,
)

_DUTY_COLON_TITLE = re.compile(
    r"^(architect|develop|build|implement|manage|ensure|optimi[sz]e|design|deploy|"
    r"create|maintain|lead|drive|coordinate)\b",
    re.IGNORECASE,
)

_STACK_CATEGORY_LINE = re.compile(
    r"^(languages?|ai protocols?|automation|environment|infrastructure)\s*:\s*",
    re.IGNORECASE,
)

_QUALIFICATIONS_SECTION_START = re.compile(r"^qualifications?\s*[:\-–—]?\s*$", re.IGNORECASE)

_CANDIDATE_SECTION_START = re.compile(
    r"^(about you|your profile|candidate profile)\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

_BRING_SECTION_START = re.compile(
    r"^what you.?ll bring\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

_NICE_TO_HAVE_START = re.compile(r"^nice to have\b", re.IGNORECASE)

_ROLE_SECTION_START = re.compile(r"^the role\b", re.IGNORECASE)

_DUTY_SECTION_START = re.compile(
    r"^(?:job\s+)?(?:key\s+)?responsibilities\b",
    re.IGNORECASE,
)

_JOB_REQUIREMENTS_START = re.compile(
    r"^job\s+requirements\b",
    re.IGNORECASE,
)

_JOB_BENEFITS_START = re.compile(
    r"^job\s+benefits\b",
    re.IGNORECASE,
)

_JOB_DESCRIPTION_START = re.compile(
    r"^job\s+description\b",
    re.IGNORECASE,
)

_LOOKING_FOR_START = re.compile(
    r"^what we.?re looking for\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

_DOING_SECTION_START = re.compile(
    r"^what you.?ll be doing\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

_BONUS_SECTION_START = re.compile(
    r"^bonus points?\s*(if you have)?:?\s*$",
    re.IGNORECASE,
)

_SOFT_SECTION_START = re.compile(
    r"^(ideal candidate|who you are)\b",
    re.IGNORECASE,
)

_OFFER_SECTION_START = re.compile(
    r"^(what.?s on offer|benefits?|perks?|why join|what we offer)\b",
    re.IGNORECASE,
)

_REQ_SECTION_START = re.compile(
    r"^(what you.?ll need|what you will learn|"
    r"key responsibilities|about the role|graduate program)",
    re.IGNORECASE,
)

_REQUIREMENTS_ENABLE_START = re.compile(
    r"^(?:job\s+)?requirements?\s*(?:[-–—]|$)|^qualifications?\s*[-–—]",
    re.IGNORECASE,
)

_TECHNICAL_QUALIFICATION = re.compile(
    r"\b(schematics?|electronic|embedded|firmware|microcontroller|bare[- ]?metal|"
    r"c programming|\bc\b(?![+#])|memory management|data structures|pointers|"
    r"hardware bring[- ]?up|unit tests?|debugging|capstone|engineering degree|"
    r"bachelor|git|ci/?cd)\b",
    re.IGNORECASE,
)

_SOFT_SKILL_ONLY = re.compile(
    r"(ability to|willingness to|strong analytical and problem-solving|"
    r"work independently while contributing|multidisciplinary team|"
    r"curious and enthusiastic to learn new technologies|"
    r"strong (communication|interpersonal)|"
    r"excellent (communication|teamwork)|team player|self[- ]?starter|"
    r"fast learner|learn quickly|quick learner|"
    r"proactive,?\s+curious|comfortable working in complex enterprise|"
    r"enjoys working across both back-end and front-end|"
    r"confident in asking the right questions|"
    r"passionate about|enthusiastic|positive attitude|cultural fit|"
    r"right to work|eligible to work|australian citizen|visa|"
    r"written and verbal communication skills)",
    re.IGNORECASE,
)

# Personality / fit lines — belong in a cover letter, not resume keyword matching.
_COVER_LETTER_TRAIT = re.compile(
    r"^(a |an )?(strong )?(team player|developer|professional|candidate) who\b|"
    r"^someone who (enjoys|is|has|can|will)\b|"
    r"^we.?re looking for (a|an) \b|"
    r"^you are (a|an) \b|"
    r"^ideal candidate (is|who)\b",
    re.IGNORECASE,
)

_TECH_SIGNAL = re.compile(
    r"\b(experience|skill|degree|bachelor|master|graduate|diploma|postgraduate|"
    r"qualification|coursework|engineering|firmware|embedded|microcontroller|"
    r"bare[- ]?metal|schematics?|debugging|capstone|fundamentals|"
    r"python|java|typescript|javascript|react|node\.?js|"
    r"angular|vue|api|cloud|azure|aws|gcp|years?|llm|ai|machine learning|"
    r"data science|computer vision|image processing|video analytics|opencv|pytorch|"
    r"tensorflow|develop|build|design|deploy|testing|agile|sql|postgres|"
    r"postgresql|mongodb|mysql|junit|mockito|spring|j2ee|ejb|camel|\.net|engineer|"
    r"retrieval[- ]?augmented|\brag\b|"
    r"proficien|proficiency|required|must|full[- ]?stack|frontend|backend|"
    r"ci/?cd|git|docker|kubernetes|rest|graphql|nosql|microservices|"
    r"unit tests?|unit testing|test plans?|firmware|embedded|microcontroller|schematics?|debugging|"
    r"relational database|computer science|software|programming|code review|"
    r"version control|scrum|swagger|openapi|redux|jquery|confluence|rally|saf[eé]|"
    r"mcp\b|playwright|puppeteer|langchain|langgraph|crewai|vertex ai|json-rpc|"
    r"function calling|schema\.org|vector search|embeddings?|semantic search|"
    r"power automate|copilot(?:\s+studio)?|low[- ]?code|no[- ]?code|agentic|"
    r"automation tools?|workflow|claude|microsoft|"
    r"symfony|php|aurelia|typescript|saas|educated|degree level|"
    r"computer science|related discipline|ai coding|coding tools?|"
    r"data structures|algorithms|c\+\+|c#|golang|oop|object[- ]oriented|design patterns|system design|software development|science)\b",
    re.IGNORECASE,
)

_COVER_LETTER_FIT_LINE = re.compile(
    r"^(product minded|enthusiastic about fully remote|self[- ]directed in|"
    r"speaking up,?\s+asking questions|contributing ideas from)\b",
    re.IGNORECASE,
)

_BENEFITS_SECTION_START = re.compile(
    r"^(benefits?|perks?|culture|our culture|life at|why join|"
    r"moose magic|employee benefits|reward|compensation)\b",
    re.IGNORECASE,
)


def _is_soft_skill_only(text: str) -> bool:
    t = text.strip()
    if _TECHNICAL_QUALIFICATION.search(t):
        return False
    if _TECH_SIGNAL.search(t) and not _COVER_LETTER_TRAIT.search(t):
        # e.g. "Java microservices" — technical even if long
        if re.search(
            r"\b(junit|mockito|spring|j2ee|python|react|postgres|aws|azure|docker|kubernetes)\b",
            t,
            re.I,
        ):
            return False
    return bool(_SOFT_SKILL_ONLY.search(t))


def is_cover_letter_trait(text: str) -> bool:
    """Traits normally written in a cover letter, not on a resume."""
    t = text.strip()
    if len(t) < 15:
        return False
    if re.search(r"\b(ai coding tools?|using ai coding)\b", t, re.I):
        return False
    if _COVER_LETTER_FIT_LINE.search(t):
        return True
    if _COVER_LETTER_TRAIT.search(t):
        return True
    if _is_soft_skill_only(t):
        return True
    return False


def _infer_priority(text: str, *, section_preferred: bool = False) -> str:
    if section_preferred:
        return "preferred"
    if re.search(
        r"\b(bonus|not essential|nice to have|desirable|optional|preferred but)\b",
        text,
        re.I,
    ):
        return "preferred"
    return "required"


def _make_bullet(text: str, *, priority: str | None = None, section_preferred: bool = False) -> dict:
    t = text.strip()
    return {
        "text": t,
        "priority": priority or _infer_priority(t, section_preferred=section_preferred),
    }


_ROLE_DUTY_START = re.compile(
    r"^(design,?\s+develop|develop,?\s+and|maintain scalable|work across|work closely|"
    r"support deployments|build a strong understanding|contribute to integration|"
    r"assist in moving|deliver high|ensure |provide |participate in)\b",
    re.IGNORECASE,
)

_RESUME_SKILL_LINE = re.compile(
    r"\b(experience with|experience in|hands-on experience|proven commercial|"
    r"strong (commercial )?experience|background in|exposure to|familiarity with|"
    r"solid core|database experience|cloud experience|front-end development experience|"
    r"proficien|proficiency in|skilled in|knowledge of|foundational knowledge|"
    r"understanding of|understandings of|confident operating|previous experience|"
    r"working across the full stack|degree in|postgraduate|qualification|"
    r"coursework in|demonstrated capability|academic focus|capability or academic)\b",
    re.IGNORECASE,
)

_INCOMPLETE_LINE = re.compile(
    r"^(deep understanding of|and struts|information technology|or a related field)\s*\.?$",
    re.IGNORECASE,
)

_SKILL_LIST_LINE = re.compile(
    r"\b(java|react|spring|api|ci/?cd|python|typescript|node)\b",
    re.IGNORECASE,
)


def _looks_like_comma_skill_tags(line: str) -> bool:
    """True for 'Java, React, Spring Boot' — not for long prose with commas."""
    if "," not in line or len(line) > 100:
        return False
    if re.search(r"^(interest in|technical aptitude)\b", line.strip(), re.I):
        return False
    if re.search(r"^experience with\s*:", line.strip(), re.I):
        return False
    if re.search(
        r"\b(demonstrated|academic focus|e\.g\.|for example|qualification|coursework|"
        r"familiarity with|proficiency in|foundational knowledge|frameworks?\s*\(|"
        r"experience|degree|understanding|confident|proven|working across|"
        r"environments|protocols|commercial)\b",
        line,
        re.I,
    ):
        return False
    if "(" in line and ")" in line:
        return False
    parts = [p.strip() for p in line.split(",") if p.strip()]
    return len(parts) >= 3 and sum(1 for p in parts if _TECH_SIGNAL.search(p)) >= 2


def _is_role_duty(text: str) -> bool:
    """Job responsibility bullets (The role) — not resume keyword skills."""
    t = text.strip()
    if _ROLE_DUTY_START.search(t):
        return True
    if re.search(
        r"\b(full sdlc|software development life cycle|delivery teams|stakeholders|"
        r"ongoing improvement initiatives|existing technology stack|moving services from test)\b",
        t,
        re.IGNORECASE,
    ):
        return True
    return False


def _is_resume_skill_requirement(text: str) -> bool:
    """Hireable stack/skill lines — what belongs on a resume."""
    t = text.strip()
    if _is_role_duty(t):
        return False
    if _RESUME_SKILL_LINE.search(t):
        return True
    if re.search(
        r"\b(junit|mockito|jmockit|j2ee|ejb|struts|spring framework|spring boot|"
        r"apache camel|reactjs|\breact\b|postgresql|swagger|openapi|oauth|jwt|"
        r"jenkins|bitbucket|bamboo|ci/?cd|terraform|ansible|cloudwatch|amazon\s+eks|\beks\b|"
        r"next\.?js|helm|gitops|devops|banking|financial services|enterprise|"
        r"pytorch|tensorflow|opencv|computer vision|image processing|video analytics|"
        r"\bpython\b|version control|git\b|scikit-learn|nlp|postgraduate|masters?|"
        r"graduate diploma|data science|artificial intelligence|mcp\b|playwright|"
        r"puppeteer|langchain|langgraph|vertex ai|gcp|google cloud|mysql|"
        r"\brag\b|vector search|semantic search|embeddings?|"
        r"embedded linux|bare[- ]?metal|firmware|microcontroller|"
        r"c programming|\bc\b(?![+#])|schematics?|unit tests?|unit testing|test plans?|debugging|"
        r"memory management|data structures|pointers|hardware bring[- ]?up|"
        r"power automate|copilot(?:\s+studio)?|low[- ]?code|no[- ]?code|agentic|"
        r"automation tools?|claude|symfony|php|aurelia|typescript|ai coding|"
        r"well[- ]tested|clean code|"
        r"algorithms|c\+\+|c#|golang|oop|object[- ]oriented|design patterns|system design|software development)\b",
        t,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(educated to degree|degree level|computer science|related discipline)\b", t, re.I):
        return True
    if re.search(r"\b\d+\+?\s*years?\s+(of\s+)?experience\b", t, re.I):
        return True
    if re.search(
        r"\b(bachelor|master|degree in engineering|engineering \(|\bengineering\b)",
        t,
        re.I,
    ):
        return True
    if _TECH_SIGNAL.search(t) and re.search(r":", t):
        return True
    return False


def _is_duty_colon_line(line: str) -> bool:
    m = re.match(r"^([^:]{3,90}):\s+(.+)$", line.strip())
    if not m:
        return False
    return bool(_DUTY_COLON_TITLE.match(m.group(1).strip()))


def _parse_colon_skill_line(line: str) -> str | None:
    """'Model Context Protocol (MCP): Deep expertise...' -> scorable skill line."""
    if re.match(r"^experience with\s*:", line.strip(), re.I):
        return None
    m = re.match(r"^([^:]{3,100}):\s+(.{12,})$", line.strip())
    if not m:
        return None
    title, body = m.group(1).strip(), m.group(2).strip()
    if _is_duty_colon_line(line) or _SUBSECTION_HEADER_ONLY.match(title):
        return None
    if _STACK_CATEGORY_LINE.match(line):
        return None
    if (
        _TECH_SIGNAL.search(title)
        or _TECH_SIGNAL.search(body)
        or re.search(r"\brag\b", title + " " + body, re.I)
    ):
        return f"{title}: {body}" if len(line) <= 320 else line
    return None


def _is_real_requirement(text: str) -> bool:
    t = text.strip()
    if len(t) < 12 or len(t) > 450:
        return False
    if _SECTION_HEADER_ONLY.match(t):
        return False
    if is_cover_letter_trait(t):
        return False
    if _is_soft_skill_only(t):
        return False
    if _RECRUITMENT_ONLY.search(t) or _OFFER_LINE.search(t):
        return False
    lower = t.lower()
    for pat in _SKIP_PATTERNS:
        if pat.search(lower):
            return False
    if not _TECH_SIGNAL.search(t):
        return False
    return _is_resume_skill_requirement(t)


def _normalize_item(item: dict, *, from_llm: bool = False) -> dict | None:
    text = str(item.get("text") or "").strip()
    if len(text) < 10 or len(text) > 450:
        return None
    if is_cover_letter_trait(text) and not jd_extract._TECHNICAL_LINE.search(text):
        return None
    if requirement_knowledge.is_junk_requirement(text):
        return None
    if _RECRUITMENT_ONLY.search(text) or _OFFER_LINE.search(text):
        return None
    if not from_llm and (not _is_real_requirement(text) or not _is_resume_skill_requirement(text)):
        return None
    if from_llm and not jd_extract._is_scorable_line(text):
        # LLM items: lighter filter — must still look technical
        if not _TECH_SIGNAL.search(text) and not jd_extract._TECHNICAL_LINE.search(text):
            return None
    category = str(item.get("category") or "other").lower()
    if category not in ("skill", "experience", "education", "other"):
        category = "other"
    priority = str(item.get("priority") or "required").lower()
    if priority not in ("required", "preferred"):
        priority = "required"
    return {"text": text, "category": category, "priority": priority, "from_llm": from_llm}


def _sort_requirements(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda x: (x.get("priority") != "required", x["text"].lower()),
    )


def _normalize_jd_text(text: str) -> str:
    """Preserve line breaks for bullet/section parsing (unlike hash normalize_text)."""
    t = text.replace("\r\n", "\n").strip().lstrip("\ufeff")
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2012", "-")
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(lines)[:12000]


def _dedupe_key(text: str) -> str:
    t = text.lower().strip().rstrip(".")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"8\+8\+", "8+", t)
    return t


def _dedupe_requirements(items: list[dict]) -> list[dict]:
    """Drop junk, duplicates, and overlapping education lines."""
    filtered = [
        item
        for item in items
        if not requirement_knowledge.is_junk_requirement(str(item.get("text") or ""))
    ]
    return requirement_knowledge.dedupe_requirements(filtered)


def _cache_get(content_hash: str) -> list[dict] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT requirements_json FROM jd_requirement_cache WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    if not row:
        return None
    try:
        data = json.loads(row["requirements_json"])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return None


def _cache_set(content_hash: str, items: list[dict]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jd_requirement_cache (content_hash, requirements_json) "
            "VALUES (?, ?)",
            (content_hash, json.dumps(items, ensure_ascii=False)),
        )


def _parse_experience_with_colon_list(line: str) -> list[str]:
    """'Experience with: Power Automate, Copilot, Copilot Studio' -> separate items."""
    m = re.match(r"^experience with\s*:\s*(.+)$", line.strip(), re.I)
    if not m:
        return []
    tail = m.group(1).strip().rstrip(".")
    parts = [p.strip() for p in re.split(r",|\band\b", tail) if p.strip()]
    out: list[str] = []
    for p in parts:
        p = re.sub(r"^microsoft\s+", "", p, flags=re.I).strip()
        if len(p) < 2:
            continue
        label = p if re.search(r"\b(experience|with)\b", p, re.I) else f"Experience with {p}"
        if _TECH_SIGNAL.search(label) or re.search(
            r"\b(power automate|copilot|claude|automate|studio)\b", label, re.I
        ):
            out.append(label)
    return out


def _expand_comma_skill_list(line: str) -> list[str]:
    """'Java, React, Spring Boot, RESTful API, CI/CD' -> separate matchable items."""
    if "," not in line or len(line) < 12:
        return []
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) < 3:
        return []
    if sum(1 for p in parts if _TECH_SIGNAL.search(p)) < 2:
        return []
    generic = re.compile(
        r"^(full stack engineer|banking|financial services|information technology)$",
        re.I,
    )
    out: list[str] = []
    for p in parts:
        p = re.sub(r"^or\s+", "", p, flags=re.I).strip().rstrip(").")
        if generic.match(p) or len(p) < 2:
            continue
        label = p if re.search(r"\b(experience|with|in)\b", p, re.I) else f"Experience with {p}"
        if _TECH_SIGNAL.search(label):
            out.append(label)
    return out


def _qualification_fragment_to_skill(line: str) -> str | None:
    """Turn 'Spring Boot' or 'RESTful APIs in Java' into a scorable skill line."""
    t = line.strip()
    if not t or _INCOMPLETE_LINE.match(t):
        return None
    t = re.sub(r"^and\s+", "", t, flags=re.I).strip()
    if len(t) < 3:
        return None
    if _is_role_duty(t) or is_cover_letter_trait(t):
        return None
    if _is_resume_skill_requirement(t):
        return t
    if _TECH_SIGNAL.search(t) and len(t) <= 60:
        if re.search(r"\b(experience|with|in|apis?)\b", t, re.I):
            return t
        return f"Experience with {t}"
    return None


def _split_compound_bullet(text: str) -> list[str]:
    """Split 'React and Node.js experience' into separate matchable items."""
    t = text.strip()
    if not re.search(r"\band\b", t, re.I):
        return [t]
    # Keep lists like "Spring, EJB, and Camel" as one requirement.
    if t.count(",") >= 2:
        return [t]
    if re.search(r"\b(exposure to|experience with|background in|familiarity with)\b", t, re.I):
        return [t]
    if re.search(r"\bunit test|test plan", t, re.I):
        return [t]
    if re.search(r"\bfull stack\b|back end and react|front end\b", t, re.I):
        return [t]
    parts = re.split(r"\s+and\s+", t, flags=re.I)
    if len(parts) < 2:
        return [t]
    tech_parts = [p.strip() for p in parts if _TECH_SIGNAL.search(p)]
    if len(tech_parts) >= 2 and all(len(p) >= 8 for p in tech_parts):
        return tech_parts
    return [t]


def _infer_category(text: str) -> str:
    lower = text.lower()
    if re.search(
        r"\b(bachelor|master|degree|graduate|diploma|postgraduate|qualification|"
        r"coursework|computer science|related field)\b",
        lower,
    ):
        return "education"
    if re.search(r"\b(\d+\+?\s*years?|experience in|production|deliver|ship)\b", lower):
        return "experience"
    if re.search(
        r"\b(python|java|react|node|typescript|sql|api|cloud|agile|full[- ]?stack|"
        r"frontend|backend|git|docker|embedded|firmware|\bc\b|microcontroller|"
        r"schematics?|debugging|unit test)\b",
        lower,
    ):
        return "skill"
    return "other"


def _llm_item_is_grounded(req_text: str, jd_text: str) -> bool:
    """
    Hallucination guard: returns False if the LLM-extracted requirement's
    key tokens are largely absent from the actual JD text.
    Prevents the model from inventing requirements not in the posting.
    """
    tokens = re.findall(r"[a-zA-Z]{3,}", req_text.lower())
    stopwords = {
        "with", "and", "the", "for", "that", "this", "have", "from",
        "experience", "knowledge", "understanding", "ability", "strong",
        "solid", "related", "using", "including", "degree", "bachelor",
        "master", "field", "required", "minimum", "preferred", "least",
        "also", "will", "are", "been", "your", "our", "can", "may",
    }
    meaningful = [t for t in tokens if t not in stopwords and len(t) > 3]
    if not meaningful:
        return True  # Short/generic items pass through
    jd_lower = jd_text.lower()
    matched = sum(1 for t in meaningful if t in jd_lower)
    return (matched / len(meaningful)) >= 0.35


def parse_requirements(jd_text: str, *, max_items: int = 18) -> list[dict]:
    """Extract hireable requirements only (skills, experience, duties) — not benefits/culture."""
    text = _normalize_jd_text(jd_text)
    if not text:
        return []

    cache_key = hash_text("v15:" + normalize_text(jd_text))
    rule_items = _rule_extract_requirements(text, max_items)

    llm_items: list[dict] = []
    if settings.jd_parse_cache_enabled:
        cached = _cache_get(cache_key)
        if cached is not None:
            llm_items = cached
            for item in llm_items:
                item["from_llm"] = True
    else:
        cached = None

    if cached is None:
        raw = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract ONLY technical requirements that are EXPLICITLY STATED in this job posting. "
                        "CRITICAL RULE: Do NOT invent, infer, or add any requirements not directly written "
                        "in the text. If a skill or tool is not mentioned, do not include it. "
                        "Read these sections only: Qualifications, Requirements, Skills, "
                        "Preferred/Nice to have, Minimum Qualifications. "
                        "INCLUDE only: named languages, frameworks, databases, tools, degree requirements, "
                        "years of experience — all must appear in the text. "
                        "One requirement per distinct item. "
                        "EXCLUDE: anything not in the text, benefits, EEO statements, soft skills, marketing. "
                        'Respond with ONLY valid JSON: '
                        '{"requirements": [{"text": string, "category": "skill"|"experience"|"education"|"other", '
                        '"priority": "required"|"preferred"}]}'
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        parsed = llm.extract_json_block(raw or "")
        if parsed:
            for item in parsed.get("requirements") or []:
                if isinstance(item, dict):
                    norm = _normalize_item(item, from_llm=True)
                    if norm and _llm_item_is_grounded(norm["text"], text):
                        norm["from_llm"] = True
                        llm_items.append(norm)
        if settings.jd_parse_cache_enabled and llm_items and len(llm_items) >= 4:
            _cache_set(cache_key, llm_items)

    # Union LLM + rules; prefer rules when LLM/cache returned too few items
    if len(llm_items) < 4 and len(rule_items) >= len(llm_items):
        merged = _dedupe_requirements(rule_items + llm_items)
    else:
        merged = _dedupe_requirements(llm_items + rule_items)
    filtered: list[dict] = []
    for item in merged:
        t = str(item.get("text") or "").strip()
        if _is_duty_colon_line(t) or _is_role_duty(t):
            continue
        norm = _normalize_item(item, from_llm=item.get("from_llm", False)) if "category" in item else None
        if norm:
            filtered.append(norm)
        elif _is_real_requirement(t) and _is_resume_skill_requirement(t):
            filtered.append(
                {
                    "text": t,
                    "category": item.get("category") or _infer_category(t),
                    "priority": item.get("priority") or "required",
                }
            )
    if not filtered and rule_items:
        for item in rule_items:
            text = str(item.get("text") or "").strip()
            if _is_real_requirement(text) and _is_resume_skill_requirement(text):
                filtered.append(
                    {
                        "text": text,
                        "category": _infer_category(text),
                        "priority": "required",
                    }
                )
    merged = _sort_requirements(filtered)[:max_items]
    return requirement_knowledge.merge_semantic_clusters(merged)


def _collect_cover_letter_lines(lines: list[str]) -> list[str]:
    traits: list[str] = []
    in_soft = False
    in_job_req = False
    for ln in lines:
        if not ln:
            continue
        if _SOFT_SECTION_START.match(ln) or _CANDIDATE_SECTION_START.match(ln):
            in_soft = True
            in_job_req = False
            continue
        if _JOB_REQUIREMENTS_START.match(ln) or _LOOKING_FOR_START.match(ln):
            in_job_req = True
            in_soft = False
            continue
        if in_job_req and (
            _JOB_BENEFITS_START.match(ln)
            or _DUTY_SECTION_START.match(ln)
            or _BENEFITS_SECTION_START.match(ln)
        ):
            in_job_req = False
            continue
        if not (in_soft or in_job_req):
            continue
        cleaned = re.sub(r"^[-•*·]\s*", "", ln).strip()
        if len(cleaned) >= 15 and is_cover_letter_trait(cleaned):
            traits.append(cleaned)
    return traits


def parse_cover_letter_traits(jd_text: str, *, max_items: int = 12) -> list[str]:
    """
    Personality/fit lines (cover letter) from 'What we're looking for' or 'About You'.
    Not scored against the resume.
    """
    text = _normalize_jd_text(jd_text)
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines()]
    traits: list[str] = list(_collect_cover_letter_lines(lines))
    in_soft = False
    in_candidate = False

    for ln in lines:
        if not ln:
            continue
        if _SOFT_SECTION_START.match(ln):
            in_soft = True
            in_candidate = False
            continue
        if _CANDIDATE_SECTION_START.match(ln) or _BRING_SECTION_START.match(ln):
            in_candidate = True
            in_soft = False
            continue
        if in_soft and (
            _SKILLS_SECTION_START.match(ln)
            or _BRING_SECTION_START.match(ln)
            or _LOOKING_FOR_START.match(ln)
            or _DOING_SECTION_START.match(ln)
            or _BONUS_SECTION_START.match(ln)
            or _QUALIFICATIONS_SECTION_START.match(ln)
            or _NICE_TO_HAVE_START.match(ln)
            or _OFFER_SECTION_START.match(ln)
            or _BENEFITS_SECTION_START.match(ln)
            or _DUTY_SECTION_START.match(ln)
            or _RECRUITMENT_ONLY.search(ln)
        ):
            in_soft = False
            continue
        if in_candidate and (
            _SKILLS_SECTION_START.match(ln)
            or _BRING_SECTION_START.match(ln)
            or _QUALIFICATIONS_SECTION_START.match(ln)
            or _NICE_TO_HAVE_START.match(ln)
            or _OFFER_SECTION_START.match(ln)
            or _DUTY_SECTION_START.match(ln)
            or _RECRUITMENT_ONLY.search(ln)
        ):
            in_candidate = False
            continue

        if not (in_soft or in_candidate):
            continue

        is_bullet = ln.startswith(("-", "•", "*", "·")) or re.match(r"^\d+[\.\)]\s", ln)
        cleaned = re.sub(r"^[-•*·]\s*", "", ln)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip()
        if not cleaned or len(cleaned) < 15:
            continue
        if _RECRUITMENT_ONLY.search(cleaned) or _OFFER_LINE.search(cleaned):
            continue
        if _SECTION_HEADER_ONLY.match(cleaned):
            continue
        if is_cover_letter_trait(cleaned):
            traits.append(cleaned)

    seen: set[str] = set()
    out: list[str] = []
    for t in traits:
        key = _dedupe_key(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def _extract_skills_from_task_line(line: str) -> list[str]:
    """Pull scorable technical phrases from role-task bullets (graduate / embedded JDs)."""
    t = line.strip()
    if len(t) < 20:
        return []
    lower = t.lower()
    found: list[str] = []
    rules: list[tuple[re.Pattern, str]] = [
        (re.compile(r"embedded\s+c\s+firmware", re.I), "Embedded C firmware development"),
        (re.compile(r"\bunit tests?\b", re.I), "Unit testing and test plans"),
        (re.compile(r"embedded\s+linux", re.I), "Embedded Linux development"),
        (re.compile(r"hardware bring[- ]?up", re.I), "Hardware bring-up experience"),
        (re.compile(r"bare[- ]?metal", re.I), "Bare-metal microcontroller development"),
        (re.compile(r"microcontrollers?", re.I), "Microcontroller firmware development"),
        (re.compile(r"\bdebugging\b", re.I), "Firmware debugging"),
    ]
    for pat, label in rules:
        if pat.search(t) and label not in found:
            found.append(label)
    if re.search(r"\bc\s+programming\b|\bembedded\s+c\b", lower) and "Embedded C" not in " ".join(found):
        found.append("Embedded C programming")
    return found


def _parse_bonus_exposure_technologies(line: str) -> list[dict]:
    """'Exposure to PHP, Symfony, TypeScript or Aurelia is a bonus...' -> preferred items."""
    m = re.search(
        r"exposure to\s+(.+?)(?:\s+is a bonus|\s+but not essential|\s+is optional|$)",
        line.strip(),
        re.I,
    )
    if not m:
        return []
    tail = m.group(1).strip().rstrip(".")
    parts = re.split(r",|\s+or\s+", tail)
    out: list[dict] = []
    for p in parts:
        p = p.strip()
        if len(p) < 2:
            continue
        out.append(_make_bullet(f"Exposure to {p}", priority="preferred"))
    return out


def _parse_job_requirement_line(line: str) -> list[dict]:
    """Parse recruiter-style 'Job requirements' bullets (graduate SaaS JDs)."""
    t = line.strip()
    if len(t) < 12:
        return []
    if is_cover_letter_trait(t):
        return []
    bonus = _parse_bonus_exposure_technologies(t)
    if bonus:
        return bonus
    if re.search(r"\b(educated to degree|degree level)\b", t, re.I):
        return [_make_bullet(t, priority="required")]
    if re.search(r"\b(ai coding tools?|using ai coding|actively using ai)\b", t, re.I):
        return [_make_bullet("AI coding tools in everyday development workflow", priority="required")]
    norm = _normalize_qualification_bullet(t)
    if norm:
        return [_make_bullet(norm)]
    if _is_real_requirement(t) and _is_resume_skill_requirement(t):
        return [_make_bullet(t)]
    return []


def _normalize_qualification_bullet(line: str) -> str | None:
    """Normalize 'What we're looking for' lines into scorable requirements."""
    t = line.strip()
    t = re.sub(r"^[-•*·]\s*", "", t)
    t = re.sub(r"^\d+[\.\)]\s*", "", t).strip()
    if not t or len(t) < 12:
        return None
    if is_cover_letter_trait(t) and not _TECHNICAL_QUALIFICATION.search(t):
        return None
    if re.search(r"^bonus points?\b", t, re.I):
        return None
    # "Solid C programming fundamentals — memory management, ..."
    if re.search(r"\bc\s+programming\b", t, re.I):
        return t if len(t) <= 200 else "Solid C programming fundamentals"
    if re.search(
        r"\b(bachelor|master|educated to degree|degree level|computer science|related discipline)\b",
        t,
        re.I,
    ):
        return t
    if re.search(r"\b(ai coding tools?|actively using ai)\b", t, re.I):
        return "AI coding tools in everyday development workflow"
    if _TECHNICAL_QUALIFICATION.search(t) or _is_resume_skill_requirement(t):
        return t
    if _qualification_fragment_to_skill(t):
        return _qualification_fragment_to_skill(t)
    return None


def _rule_extract_requirements(text: str, max_items: int) -> list[dict]:
    """Generic extractor + legacy rules, deduped."""
    generic = jd_extract.extract_requirements(text, max_items=max_items)
    legacy = _fallback_requirements(text, max_items)
    return _merge_rule_lists(generic, legacy, max_items)


def _merge_rule_lists(a: list[dict], b: list[dict], max_items: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in a + b:
        key = re.sub(r"\s+", " ", str(item.get("text", "")).lower().strip())[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def _fallback_requirements(text: str, max_items: int) -> list[dict]:
    """Extract from Skills, About You, Qualifications, Nice to have — not Key Responsibilities."""
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").splitlines()]
    in_skills = False
    in_nice = False
    in_duty = False
    in_doing = False
    bullets: list[dict] = []

    for ln in lines:
        if not ln:
            continue
        if _RECRUITMENT_ONLY.search(ln):
            in_skills = False
            in_nice = False
            in_duty = False
            in_doing = False
            continue
        if _LOOKING_FOR_START.match(ln):
            in_skills = True
            in_nice = False
            in_duty = False
            in_doing = False
            continue
        if _DOING_SECTION_START.match(ln):
            in_doing = True
            in_skills = False
            in_nice = False
            in_duty = False
            continue
        if _JOB_REQUIREMENTS_START.match(ln) or _REQUIREMENTS_ENABLE_START.match(ln):
            in_skills = True
            in_nice = False
            in_duty = False
            in_doing = False
            continue
        if _JOB_BENEFITS_START.match(ln) or _JOB_DESCRIPTION_START.match(ln):
            in_skills = False
            in_nice = False
            in_duty = False
            in_doing = False
            continue
        if _BONUS_SECTION_START.match(ln):
            in_nice = True
            in_skills = False
            in_duty = False
            in_doing = False
            continue
        if (
            _BENEFITS_SECTION_START.match(ln)
            or _OFFER_SECTION_START.match(ln)
            or _SOFT_SECTION_START.match(ln)
            or _ROLE_SECTION_START.match(ln)
            or _DUTY_SECTION_START.match(ln)
        ):
            in_skills = False
            in_nice = False
            in_doing = False
            in_duty = _DUTY_SECTION_START.match(ln) is not None
            continue
        if _SKILLS_SECTION_START.match(ln) or re.match(r"^technical stack\b", ln, re.I):
            in_skills = True
            in_nice = False
            in_duty = False
            if _STACK_CATEGORY_LINE.match(ln):
                continue
            continue
        if (
            _CANDIDATE_SECTION_START.match(ln)
            or _BRING_SECTION_START.match(ln)
            or _QUALIFICATIONS_SECTION_START.match(ln)
        ):
            in_skills = True
            in_nice = False
            in_duty = False
            continue
        if _NICE_TO_HAVE_START.match(ln):
            in_nice = True
            in_skills = False
            in_duty = False
            continue
        if _REQ_SECTION_START.match(ln) and not (
            _SKILLS_SECTION_START.match(ln) or _QUALIFICATIONS_SECTION_START.match(ln)
        ):
            in_skills = False
            in_nice = False
            continue
        if _SECTION_HEADER_ONLY.match(ln):
            continue

        if in_duty:
            cleaned_duty = re.sub(r"^[-•*·]\s*", "", ln).strip()
            if re.search(r"\bai coding tools?\b", cleaned_duty, re.I):
                bullets.append(
                    _make_bullet("AI coding tools in everyday development workflow")
                )
            if re.search(r"\bwell[- ]tested|clean,?\s+well[- ]tested code\b", cleaned_duty, re.I):
                bullets.append(_make_bullet("Writing clean, well-tested code"))
            continue

        if in_doing:
            cleaned_doing = re.sub(r"^[-•*·]\s*", "", ln).strip()
            for skill in _extract_skills_from_task_line(cleaned_doing):
                bullets.append(_make_bullet(skill))
            continue

        in_scorable = in_skills or in_nice

        if in_scorable and _SUBSECTION_HEADER_ONLY.match(ln):
            continue

        if in_scorable:
            exp_list = _parse_experience_with_colon_list(ln)
            if exp_list:
                for item in exp_list:
                    bullets.append(_make_bullet(item))
                continue

        if in_skills and not in_nice:
            job_reqs = _parse_job_requirement_line(ln)
            if job_reqs:
                bullets.extend(job_reqs)
                continue
            norm_q = _normalize_qualification_bullet(ln)
            if norm_q:
                bullets.append(_make_bullet(norm_q))
                continue

        if in_scorable:
            colon_skill = _parse_colon_skill_line(ln)
            if colon_skill and _is_real_requirement(colon_skill):
                bullets.append(_make_bullet(colon_skill))
                continue

        if in_scorable and _STACK_CATEGORY_LINE.match(ln):
            tail = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if tail and _looks_like_comma_skill_tags(tail):
                for item in _expand_comma_skill_list(tail):
                    bullets.append(_make_bullet(item))
                continue
            if tail:
                for part in re.split(r",|\band\b", tail):
                    part = re.sub(r"\([^)]*\)", "", part).strip()
                    if part and _TECH_SIGNAL.search(part):
                        bullets.append(_make_bullet(f"Experience with {part}"))
            continue

        is_bullet = ln.startswith(("-", "•", "*", "·")) or re.match(r"^\d+[\.\)]\s", ln)
        cleaned = re.sub(r"^[-•*·]\s*", "", ln)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip()

        if in_scorable and _looks_like_comma_skill_tags(cleaned):
            for item in _expand_comma_skill_list(cleaned):
                bullets.append(_make_bullet(item))
            continue
        if in_scorable:
            frag = _qualification_fragment_to_skill(cleaned if is_bullet else ln)
            if frag:
                bullets.append(_make_bullet(frag))
                continue
        if is_bullet and in_scorable and _is_real_requirement(cleaned):
            bullets.append(_make_bullet(cleaned, section_preferred=in_nice))
        elif in_scorable and not is_bullet and _is_real_requirement(ln):
            bullets.append(_make_bullet(ln, section_preferred=in_nice))

    # Last resort: skill-pattern lines only (never role duties)
    if not bullets:
        for ln in lines:
            if ln.startswith(("-", "•", "*")):
                cleaned = ln.lstrip("-•* ").strip()
                if _is_real_requirement(cleaned) and _is_resume_skill_requirement(cleaned):
                    bullets.append(_make_bullet(cleaned))

    # Inline tech stacks on a single line: "experience with React, Node, PostgreSQL"
    if len(bullets) < 4 and "\n" not in text[:500]:
        for m in re.finditer(
            r"(?:experience with|proficient in|knowledge of|skills in)\s+"
            r"([A-Za-z0-9+.#/,\s-]{8,80})",
            text,
            re.I,
        ):
            frag = m.group(1).strip().rstrip(".")
            for part in re.split(r",|\band\b", frag):
                part = part.strip()
                if part and _is_real_requirement(part):
                    bullets.append(_make_bullet(part))

    seen: set[str] = set()
    out: list[dict] = []
    for entry in bullets:
        text = str(entry.get("text") or "").strip()
        base_priority = str(entry.get("priority") or "required")
        for piece in _split_compound_bullet(text):
            if not _is_resume_skill_requirement(piece):
                continue
            key = _dedupe_key(piece)
            if key in seen:
                continue
            seen.add(key)
            priority = _infer_priority(piece) if base_priority == "required" else base_priority
            if in_nice:
                priority = "preferred"
            out.append(
                {
                    "text": piece,
                    "category": _infer_category(piece),
                    "priority": priority,
                }
            )
            if len(out) >= max_items:
                return out
    return out
