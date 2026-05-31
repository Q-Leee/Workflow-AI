from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: str = "./data"
    upload_dir: str = "./data/uploads"
    chroma_path: str = "./data/chroma"
    sqlite_path: str = "./data/workflow.db"
    chroma_collection: str = "workflow_ai_chunks"

    embedding_provider: str = "local"
    embedding_model_local: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_multiplier: int = 6
    hybrid_pool_limit: int = 400

    hybrid_search_enabled: bool = True
    page_context_expand: bool = True
    page_context_max_chars: int = 2800

    chunk_size: int = 800
    chunk_overlap: int = 180

    query_top_k_default: int = 10
    query_top_k_list: int = 18
    answer_min_relevance: float = 0.28
    answer_two_pass_enabled: bool = True

    match_requirement_top_k: int = 18
    match_retrieval_stage1_top_k: int = 24
    match_reuse_indexed_docs: bool = True
    match_llm_summary_only: bool = True
    # hybrid = BM25+vector retrieval per requirement, then LLM confirms/adjusts
    # retrieval = retrieval only; llm_judge = LLM-only batch (no retrieval pre-pass)
    match_scoring_mode: str = "hybrid"
    match_use_llm_judge: bool = True
    jd_parse_use_rag_context: bool = True
    jd_parse_min_items_for_cache: int = 4
    match_judge_batch_size: int = 8
    match_resume_context_max_chars: int = 14000
    match_strength_partial_min_score: float = 45.0
    jd_parse_cache_enabled: bool = True
    ollama_temperature: float = 0.0

    ollama_chat_model: str = "qwen2.5-coder:7b"
    llm_enabled: bool = True

    # ── Meeting summarization ──────────────────────────────────────────
    # Characters per chunk fed to the LLM in a single call.
    # Increase if your model supports a larger context window.
    meeting_chunk_chars: int = 6000
    # Overlap between consecutive chunks to preserve context at boundaries.
    meeting_chunk_overlap_chars: int = 400
    # Temperature for the meeting LLM calls (0.0 = deterministic JSON output).
    meeting_temperature: float = 0.0
    # Optional: override the model used specifically for meeting summarization.
    # Leave empty to reuse ollama_chat_model.
    meeting_chat_model: str = ""

    jwt_secret: str = "change-me-in-production-use-env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


settings = Settings()
