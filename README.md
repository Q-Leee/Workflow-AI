# Workflow-AI

An enterprise-grade **Talent Acquisition, Candidate Matching, and Document Q&A Platform** powered by **Hybrid Retrieval-Augmented Generation (RAG)**. 

Workflow-AI dynamically parses, indexes, and matches candidate profiles (resumes) against job requirements (JDs), performs precise document Q&A, and automates meeting/interview transcript summarization using a state-of-the-art local AI pipeline.

---

## 🚀 Key Architectural Features

### 1. Hybrid Dense-Sparse Retrieval
* **Dense Vector Search**: Embeds document chunks using `sentence-transformers/all-MiniLM-L6-v2` (or `nomic-embed-text` via Ollama) and indexes them in **ChromaDB**.
* **Sparse Lexical Search**: Uses a **BM25 algorithm** (`rank-bm25`) to catch exact keyword matches (like specific technologies, tool names, or certificates).
* **Reciprocal Rank Fusion**: Merges both dense and sparse scores dynamically to deliver highly relevant context matches.

### 2. Multi-Stage Reranking & Page Context Expansion
* **Cross-Encoder Reranking**: Re-orders retrieved chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2` to filter out low-relevance results.
* **Page-Level Reconstruction**: When a specific chunk matches, the system dynamically pulls and merges neighboring chunks from the same page (`page_context_expand`). This prevents the LLM from losing boundary context and keeps text structures (like tables and lists) intact.

### 3. Dynamic JIT (Just-In-Time) URL Scraping
* When a user submits a question containing a URL (e.g., a Job Description link), the RAG pipeline intercepts it.
* **Smart Cache**: Checks ChromaDB for cached scraped results.
* **Dynamic Crawling**: On cache miss, it scrapes the web page, converts the layout to clean markdown, chunks it, and ingests it into ChromaDB on-the-fly, making external web references instantly searchable.

### 4. Two-Pass cited Generation (Anti-Hallucination)
To ensure absolute accuracy for compliance, hiring metrics, and legal reviews, the synthesis pipeline utilizes a two-pass architecture:
1. **Fact Extraction Pass**: The LLM extracts atomic facts from retrieved context with precise source index citations (e.g., `[1]`, `[2]`).
2. **Narrative Synthesis Pass**: The LLM compiles the isolated facts into a coherent final answer, carrying over the source citations, guaranteeing 100% factual alignment.

### 5. Talent & JD Matching Engine (LLM-as-a-Judge)
* **Requirement Extraction**: Parses complex JDs and normalizes them into modular requirement schemas (Skills, Experience, Certifications).
* **Evaluation Pipeline**: Retrieves candidate resume facts using hybrid search, and runs an **LLM-as-a-Judge** scoring engine to evaluate matches and strength scores with granular explanations.

### 6. Meeting Transcription & Interview Summarizer
* Features a robust summarizer that splits long transcripts into optimal sliding windows, processes them in parallel, and merges them into structured summaries, key takeaways, and action items.

---

## 🛠 Tech Stack

### Backend
* **Core Framework**: FastAPI, Uvicorn, Pydantic (v2)
* **Vector Database**: ChromaDB
* **Relational Database**: SQLite
* **Embeddings & Reranker**: Sentence-Transformers, Rank-BM25, MS-Marco Cross-Encoder
* **LLM Engine**: Ollama (supports local models like `qwen2.5-coder:7b` or `llama3`)
* **Document Parsers**: PyPDF, Python-docx, OpenPyxl (for PDF, Word, Excel extraction)

### Frontend
* **Core Stack**: React, Vite, TypeScript
* **State & Routing**: Standard clean UI components with custom CSS design tokens

---

## 📁 Repository Structure

```directory
Workflow-AI/
├── backend/
│   ├── app/
│   │   ├── config.py           # Settings Config using Pydantic Settings
│   │   ├── deps.py             # Dependency Injections
│   │   ├── main.py             # FastAPI App, Routes, JWT Security
│   │   ├── db/                 # SQLite Schema and Connections
│   │   ├── models/             # Pydantic Schemas & DB Models
│   │   └── services/           # Core AI Engine (RAG, Match, Meeting, etc.)
│   ├── data/                   # Chroma DB vector store & SQLite store
│   ├── scripts/                # Evaluation & seeding scripts
│   ├── requirements.txt        # Python backend dependencies
│   └── run.ps1 / run.bat       # Backend run scripts
└── frontend/
    ├── src/
    │   ├── pages/              # LoginPage, MeetingPage, PdfQaPage, ResumeMatchPage
    │   ├── components/         # Reusable UI Components
    │   ├── api/                # API Client Methods
    │   ├── auth/               # User Authentication Context
    │   └── styles.css          # Modern dark-theme glassmorphism styling
    ├── package.json
    └── vite.config.ts
```

---

## ⚙️ Getting Started

### Prerequisites
* **Python 3.10+**
* **Node.js 18+**
* **Ollama** installed and running locally.

### 1. Setup Backend

1. Install Ollama and pull the models:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text
   ```

2. Navigate to backend and create environment file:
   ```bash
   cd backend
   cp .env.example .env
   ```

3. Run the backend bootstrapper (automatically creates `.venv`, installs packages, and starts server):
   ```powershell
   # Windows PowerShell
   .\run.ps1
   ```
   The backend API will run on `http://127.0.0.1:8000` (docs available at `/docs`).

### 2. Setup Frontend

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run Vite Development Server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

---

## 🔒 Security & Auth
* Authenticated routes are secured using **JWT tokens (RS256/HS256)**.
* Multi-tenant data isolation: Vector search and DB queries are strictly scoped using `user_id` context filters.
