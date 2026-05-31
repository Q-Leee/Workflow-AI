import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.config import settings
from app.db.database import init_db
from app.deps import get_current_user
from app.models.schemas import (
    DocumentOut,
    DocumentUploadResponse,
    LoginRequest,
    MatchResponse,
    MeetingSummaryResponse,
    QueryHistoryOut,
    QueryRequest,
    QueryResponse,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from app.services import audit, auth_service, match_service, meeting_service, rag
from app.services.content_hash import hash_bytes, hash_text
from app.services.document_extract import SUPPORTED_EXTENSIONS

RESUME_EXTENSIONS = {".pdf", ".docx"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WorkFlow AI", version="0.3.0")

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s %s -> %s %.1fms", rid, request.method, request.url.path, response.status_code, ms)
    return response


@app.on_event("startup")
def startup() -> None:
    init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_answer": "skills-fallback-v2",
        "ollama_model": settings.ollama_chat_model,
    }


@app.post("/auth/register", response_model=TokenResponse)
def register(body: RegisterRequest):
    try:
        user = auth_service.create_user(body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    token = auth_service.create_access_token(user["id"], user["email"])
    audit.log_action(user_id=user["id"], action="register", detail={"email": user["email"]})
    return TokenResponse(
        access_token=token,
        user=UserOut(id=user["id"], email=user["email"]),
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = auth_service.authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth_service.create_access_token(user["id"], user["email"])
    audit.log_action(user_id=user["id"], action="login", detail={})
    return TokenResponse(
        access_token=token,
        user=UserOut(id=user["id"], email=user["email"]),
    )


@app.get("/auth/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    return UserOut(id=user["id"], email=user["email"])


def _save_upload(raw: bytes, suffix: str) -> Path:
    doc_dir = Path(settings.upload_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / f"upload_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(raw)
    return path


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(default="pdf"),
    user: dict = Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    if doc_type not in ("pdf", "resume", "jd", "meeting"):
        raise HTTPException(status_code=400, detail="Invalid doc_type")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    tmp_path = _save_upload(raw, ext)

    try:
        doc_id, n_chunks = rag.ingest_document_file(
            file_path=tmp_path,
            original_filename=safe_name,
            user_id=user["id"],
            doc_type=doc_type,
        )
    except ValueError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        logger.exception("Ingest failed")
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document") from None

    final_path = Path(settings.upload_dir) / f"{doc_id}_{safe_name}"
    try:
        tmp_path.rename(final_path)
    except OSError:
        pass

    auth_service.register_document(
        user_id=user["id"],
        document_id=doc_id,
        filename=safe_name,
        doc_type=doc_type,
        chunks=n_chunks,
    )
    audit.log_action(
        user_id=user["id"],
        action="document_upload",
        detail={"document_id": doc_id, "filename": safe_name, "doc_type": doc_type, "chunks": n_chunks},
    )
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        doc_type=doc_type,
        chunks_indexed=n_chunks,
    )


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(user: dict = Depends(get_current_user)):
    rows = auth_service.list_documents(user["id"])
    return [DocumentOut(**r) for r in rows]


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, user: dict = Depends(get_current_user)):
    if not auth_service.delete_document(user_id=user["id"], document_id=document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    audit.log_action(user_id=user["id"], action="document_delete", detail={"document_id": document_id})
    return {"ok": True}


@app.get("/queries/history", response_model=list[QueryHistoryOut])
def query_history(
    document_id: str | None = None,
    limit: int = 20,
    user: dict = Depends(get_current_user),
):
    rows = auth_service.list_query_history(
        user_id=user["id"],
        document_id=document_id,
        limit=min(limit, 50),
    )
    return [QueryHistoryOut(**r) for r in rows]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, user: dict = Depends(get_current_user)):
    if req.document_id and not auth_service.user_owns_document(user["id"], req.document_id):
        raise HTTPException(status_code=403, detail="Document not found")

    do_rerank = settings.rerank_enabled if req.rerank is None else req.rerank
    use_llm = settings.llm_enabled if req.use_llm is None else req.use_llm

    sources = rag.retrieve_and_rerank(
        user_id=user["id"],
        question=req.question.strip(),
        document_id=req.document_id,
        filename=req.filename,
        doc_type=None,
        page_min=req.page_min,
        page_max=req.page_max,
        top_k=req.top_k,
        do_rerank=do_rerank,
    )

    answer = None
    if use_llm:
        if sources:
            answer = rag.generate_answer_ollama(req.question.strip(), sources)
            if answer == rag.LOW_CONFIDENCE_ANSWER and rag._is_skill_question(req.question.strip()):
                fb = rag._answer_skills_from_sources(sources)
                if fb:
                    answer = fb
        else:
            answer = rag.LOW_CONFIDENCE_ANSWER

    auth_service.save_query_history(
        user_id=user["id"],
        document_id=req.document_id,
        question=req.question.strip(),
        answer=answer,
        source_count=len(sources),
    )
    audit.log_action(
        user_id=user["id"],
        action="query",
        detail={"document_id": req.document_id, "source_count": len(sources), "use_llm": use_llm},
    )
    return QueryResponse(question=req.question, answer=answer, sources=sources)


@app.post("/match", response_model=MatchResponse)
async def match_resume_jd(
    resume: UploadFile = File(...),
    jd: UploadFile | None = File(default=None),
    jd_text: str | None = Form(default=None),
    use_llm: bool = Form(default=True),
    user: dict = Depends(get_current_user),
):
    resume_name = Path(resume.filename).name if resume.filename else ""
    resume_ext = Path(resume_name).suffix.lower()
    if resume_ext not in RESUME_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Resume must be PDF or Word (.docx). Supported: {', '.join(sorted(RESUME_EXTENSIONS))}",
        )

    resume_raw = await resume.read()
    if not resume_raw:
        raise HTTPException(status_code=400, detail="Empty resume file")

    jd_plain = (jd_text or "").strip()
    jd_path: Path | None = None
    jd_filename = "job_description.txt"

    if jd and jd.filename:
        if not jd.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="JD file must be a PDF")
        jd_raw = await jd.read()
        if not jd_raw:
            raise HTTPException(status_code=400, detail="Empty JD file")
        jd_path = _save_upload(jd_raw, ".pdf")
        jd_filename = Path(jd.filename).name
    elif not jd_plain:
        raise HTTPException(
            status_code=400,
            detail="Provide job description as pasted text (jd_text) or upload a JD PDF",
        )

    resume_path = _save_upload(resume_raw, resume_ext)
    resume_hash = hash_bytes(resume_raw)

    if jd_plain:
        jd_hash = hash_text(jd_plain)
    else:
        jd_hash = hash_bytes(jd_raw)

    try:
        result = match_service.match_resume_to_jd(
            user_id=user["id"],
            resume_path=resume_path,
            resume_filename=resume_name,
            resume_content_hash=resume_hash,
            jd_path=jd_path,
            jd_filename=jd_filename,
            jd_text=jd_plain or None,
            jd_content_hash=jd_hash,
            use_llm=use_llm,
        )
    except Exception:
        logger.exception("Match failed")
        raise HTTPException(status_code=500, detail="Match failed") from None

    audit.log_action(user_id=user["id"], action="resume_match", detail={"score": result.overall_score})
    return result


@app.post("/meetings/summarize", response_model=MeetingSummaryResponse)
async def summarize_meeting(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    use_llm: bool = Form(default=True),
    user: dict = Depends(get_current_user),
):
    content = (text or "").strip()

    if file and file.filename:
        safe = Path(file.filename).name
        ext = Path(safe).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        path = _save_upload(raw, ext)
        try:
            content = meeting_service.extract_text_from_upload(path, safe)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    if not content:
        raise HTTPException(status_code=400, detail="Provide file or text")

    result = meeting_service.summarize_meeting_text(content, use_llm=use_llm)
    audit.log_action(
        user_id=user["id"],
        action="meeting_summarize",
        detail={"length": result.raw_transcript_length, "actions": len(result.action_items)},
    )
    return result
