import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import LoadingPanel from "../components/LoadingPanel";

type LoadingKind = "upload" | "query" | null;

const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".xlsm", ".txt"] as const;
const FILE_ACCEPT =
  ".pdf,.docx,.xlsx,.xlsm,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/plain";

function fileExtension(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function isSupportedFile(file: File): boolean {
  return SUPPORTED_EXTENSIONS.includes(fileExtension(file.name) as (typeof SUPPORTED_EXTENSIONS)[number]);
}

function formatLabel(name: string): string {
  const ext = fileExtension(name);
  if (ext === ".pdf") return "PDF";
  if (ext === ".docx") return "Word";
  if (ext === ".xlsx" || ext === ".xlsm") return "Excel";
  if (ext === ".txt") return "Text";
  return "Document";
}

export default function PdfQaPage() {
  const [file, setFile] = useState<File | null>(null);
  const [docId, setDocId] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<{ text: string; page: number; filename: string }[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState<LoadingKind>(null);
  const [hasQueried, setHasQueried] = useState(false);
  const [topK, setTopK] = useState(12);
  const resultsRef = useRef<HTMLElement>(null);
  const [history, setHistory] = useState<
    { id: number; question: string; answer: string | null; created_at: string }[]
  >([]);

  const loadHistory = async (id?: string) => {
    try {
      const rows = await api.queryHistory(id);
      setHistory(rows);
    } catch {
      setHistory([]);
    }
  };

  useEffect(() => {
    if (docId) void loadHistory(docId);
  }, [docId]);

  const doUpload = async (picked: File) => {
    if (!isSupportedFile(picked)) {
      setStatus(`Unsupported file. Use: ${SUPPORTED_EXTENSIONS.join(", ")}`);
      return;
    }
    setBusy(true);
    setLoading("upload");
    setStatus("");
    setHasQueried(false);
    setAnswer(null);
    setSources([]);
    try {
      const res = await api.uploadDocument(picked, "pdf");
      setDocId(res.document_id);
      const kind = formatLabel(picked.name);
      if (res.chunks_indexed === 0) {
        setStatus(
          `Uploaded ${kind} but no text found (scanned PDF or empty sheet?). Try another file.`
        );
      } else {
        setStatus(`✓ ${kind} ready — ${res.chunks_indexed} chunks indexed. You can click Ask now.`);
      }
    } catch (err) {
      setDocId("");
      setStatus(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
      setLoading(null);
    }
  };

  const onFilePicked = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0] ?? null;
    setFile(picked);
    if (picked) void doUpload(picked);
    e.target.value = "";
  };

  const upload = async (e: FormEvent) => {
    e.preventDefault();
    if (file) await doUpload(file);
  };

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim()) {
      setStatus("Type a question first.");
      return;
    }
    if (!docId) {
      setStatus("Choose a document above first — wait until you see “Ready”.");
      return;
    }
    setBusy(true);
    setLoading("query");
    setStatus("");
    setAnswer(null);
    setSources([]);
    setHasQueried(false);
    try {
      const res = await api.query({
        question,
        document_id: docId,
        top_k: topK,
        use_llm: true,
      });
      setAnswer(res.answer);
      setSources(res.sources);
      setHasQueried(true);
      setStatus(res.sources.length ? "" : "No matching text found — try another question.");
      void loadHistory(docId);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Query failed");
      setHasQueried(true);
    } finally {
      setBusy(false);
      setLoading(null);
    }
  };

  return (
    <div className="stack">
      <section className="card">
        <h2>1. Upload document</h2>
        <p className="muted">
          PDF · Word (.docx) · Excel (.xlsx, .xlsm) · plain text (.txt). File uploads automatically
          — wait for “Ready” before Ask.
        </p>
        <form onSubmit={upload} className="row">
          <input
            type="file"
            accept={FILE_ACCEPT}
            onChange={onFilePicked}
            disabled={busy}
          />
          <button type="submit" className="btn primary" disabled={!file || busy}>
            {busy && loading === "upload" ? "Uploading…" : "Re-upload"}
          </button>
        </form>
        {loading === "upload" && (
          <LoadingPanel
            label="Indexing document"
            sublabel="Extracting text and building vector index…"
          />
        )}
        {docId ? (
          <p className="success">
            Document linked ·{" "}
            <button type="button" className="btn link" onClick={() => void loadHistory(docId)}>
              refresh history
            </button>
          </p>
        ) : (
          <p className="hint">↑ Pick PDF, Word, or Excel to enable questions</p>
        )}
      </section>

      <section className="card">
        <h2>2. Ask a question (RAG)</h2>
        <form onSubmit={ask} className="stack">
          <label className="row-label">
            How many excerpts to search (more = more jobs listed, slower)
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={12}>12 (recommended for lists)</option>
              <option value={20}>20</option>
            </select>
          </label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What programming languages are mentioned?"
            rows={3}
          />
          <button type="submit" className="btn primary" disabled={busy}>
            {busy && loading === "query" ? "Working…" : "Ask"}
          </button>
          {loading === "query" && (
            <LoadingPanel
              label="Answering"
              sublabel="Searching sources and generating answer with Ollama…"
            />
          )}
        </form>
      </section>

      {status && (
        <p
          className={
            status.startsWith("✓")
              ? "success"
              : status.includes("fail") ||
                  status.includes("first") ||
                  status.includes("Type") ||
                  status.includes("Unsupported")
                ? "error"
                : "muted"
          }
        >
          {status}
        </p>
      )}

      {history.length > 0 && (
        <section className="card">
          <h3>Recent questions (this document)</h3>
          <ul className="history-list">
            {history.slice(0, 8).map((h) => (
              <li key={h.id}>
                <strong>{h.question}</strong>
                {h.answer && <p className="muted small">{h.answer.slice(0, 200)}…</p>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {(hasQueried || sources.length > 0 || answer) && (
        <section ref={resultsRef} className="stack results">
          <h2 className="results-title">3. Results</h2>

          <section className="card highlight">
            <h3>Answer (LLM)</h3>
            {answer ? (
              <p className="answer-text">{answer}</p>
            ) : (
              <p className="muted">
                No LLM answer — start Ollama (<code>ollama serve</code>) or check sources below.
              </p>
            )}
          </section>

          {sources.length > 0 ? (
            <section className="card">
              <h3>Sources ({sources.length})</h3>
              <ul className="sources">
                {sources.map((s, i) => (
                  <li key={i}>
                    <strong>
                      [{i + 1}] {s.filename} · page {s.page}
                    </strong>
                    <p>{s.text.length > 500 ? `${s.text.slice(0, 500)}…` : s.text}</p>
                  </li>
                ))}
              </ul>
            </section>
          ) : hasQueried ? (
            <section className="card">
              <p className="muted">No excerpts found. Re-upload the document and try again.</p>
            </section>
          ) : null}
        </section>
      )}
    </div>
  );
}
