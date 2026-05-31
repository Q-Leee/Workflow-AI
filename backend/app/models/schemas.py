from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserOut(BaseModel):
    id: str
    email: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    doc_type: str
    chunks_indexed: int


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int
    chunk_index: int
    section: str = "general"
    vector_score: float | None = None
    rerank_score: float | None = None
    text: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What programming languages are on the resume?",
                    "document_id": "paste-document-id-from-upload-response",
                    "top_k": 10,
                    "use_llm": True,
                }
            ]
        }
    )

    question: str = Field(..., min_length=1)
    document_id: str | None = None
    filename: str | None = None
    page_min: int | None = Field(default=None, ge=0)
    page_max: int | None = Field(default=None, ge=0)
    top_k: int = Field(default=10, ge=1, le=30)
    rerank: bool | None = None
    use_llm: bool | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str | None = None
    sources: list[SourceChunk]


class MatchedPair(BaseModel):
    jd_excerpt: str
    resume_excerpt: str
    relevance_score: float | None = None


class RequirementMatch(BaseModel):
    requirement: str
    category: str = "other"
    priority: str = "required"
    status: str  # met | partial | missing
    resume_citation: str | None = None
    resume_excerpt: str | None = None
    explanation: str = ""
    score: float = 0.0


class MatchResponse(BaseModel):
    resume_document_id: str
    jd_document_id: str
    overall_score: float
    score_note: str | None = None
    summary: str | None = None
    strengths: list[str] = []
    gaps: list[str] = []
    cover_letter_topics: list[str] = Field(
        default_factory=list,
        description="JD personality/fit lines — for cover letter, not scored on resume",
    )
    requirement_matches: list[RequirementMatch] = []
    matched_pairs: list[MatchedPair] = []
    resume_sources: list[SourceChunk] = []
    jd_sources: list[SourceChunk] = []


class DocumentOut(BaseModel):
    id: str
    filename: str
    doc_type: str
    chunks_indexed: int
    created_at: str


class QueryHistoryOut(BaseModel):
    id: int
    document_id: str | None
    question: str
    answer: str | None
    source_count: int
    created_at: str


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due: str | None = None


class MeetingSummaryResponse(BaseModel):
    summary: str
    action_items: list[ActionItem]
    raw_transcript_length: int
