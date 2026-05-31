import { FormEvent, useState } from "react";
import { api } from "../api/client";
import LoadingPanel from "../components/LoadingPanel";

export default function ResumeMatchPage() {
  const [resume, setResume] = useState<File | null>(null);
  const [jdText, setJdText] = useState("");
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [useJdFile, setUseJdFile] = useState(false);
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.match>> | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const hasJd = useJdFile ? !!jdFile : jdText.trim().length > 20;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!resume || !hasJd) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const res = await api.match({
        resume,
        jdText: useJdFile ? undefined : jdText,
        jdFile: useJdFile ? jdFile ?? undefined : undefined,
        useLlm: true,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Match failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <section className="card">
        <h2>Resume ↔ Job Description</h2>
        <p className="muted">
          Resume: PDF or Word (.docx). JD: paste from LinkedIn/Indeed (recommended) or optional PDF.
        </p>
        <form onSubmit={submit} className="stack">
          <label>
            Resume (PDF or Word)
            <input
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(e) => setResume(e.target.files?.[0] ?? null)}
            />
          </label>

          <label className="row-label">
            <span>JD input mode</span>
            <select
              value={useJdFile ? "file" : "paste"}
              onChange={(e) => setUseJdFile(e.target.value === "file")}
            >
              <option value="paste">Paste text (recommended)</option>
              <option value="file">Upload PDF</option>
            </select>
          </label>

          {useJdFile ? (
            <label>
              Job description (PDF)
              <input type="file" accept=".pdf" onChange={(e) => setJdFile(e.target.files?.[0] ?? null)} />
            </label>
          ) : (
            <label>
              Job description (paste)
              <textarea
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                rows={10}
                placeholder="Paste the full job posting here…"
              />
            </label>
          )}

          <button type="submit" className="btn primary" disabled={!resume || !hasJd || busy}>
            {busy ? "Matching…" : "Run match"}
          </button>
          {busy && (
            <LoadingPanel
              label="Matching resume to job description"
              sublabel="Indexing documents, vector search, and LLM analysis…"
            />
          )}
        </form>
        {!useJdFile && jdText.trim().length > 0 && jdText.trim().length <= 20 && (
          <p className="hint">Paste a bit more of the job description (at least ~20 characters).</p>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="card highlight">
          <h3>Score: {result.overall_score}%</h3>
          {result.score_note && <p className="muted small">{result.score_note}</p>}
          {result.summary && <p>{result.summary}</p>}
          {result.strengths.length > 0 && (
            <>
              <h4>Strengths</h4>
              <ul>
                {result.strengths.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </>
          )}
          {result.gaps.length > 0 && (
            <>
              <h4>Gaps</h4>
              <ul>
                {result.gaps.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </>
          )}
          {result.cover_letter_topics && result.cover_letter_topics.length > 0 && (
            <section className="cover-letter-topics">
              <h4>Cover letter topics (not scored)</h4>
              <p className="muted small">
                These come from &quot;What we&apos;re looking for&quot; style sections. They belong in a
                cover letter, not on a resume — so they do not affect your % score.
              </p>
              <ul>
                {result.cover_letter_topics.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
            </section>
          )}
          {result.requirement_matches?.length > 0 && (
            <section className="req-table-wrap">
              <h4>Resume skills match</h4>
              <p className="muted small">
                Technical requirements from the JD, matched against your resume. Click a row for
                details.
              </p>
              <table className="req-table req-table-single">
                <thead>
                  <tr>
                    <th>Requirement — click to expand</th>
                  </tr>
                </thead>
                <tbody>
                  {result.requirement_matches.map((r, i) => (
                    <tr key={i} className="req-row">
                      <td className="req-cell">
                        <details className="req-details">
                          <summary className="req-summary">
                            <span className="req-caret" aria-hidden />
                            <div className="req-summary-content">
                              <div className="req-summary-text">{r.requirement}</div>
                              <div className="req-summary-meta">
                                <span className={`badge badge-${r.status}`}>{r.status}</span>
                                <span className="req-score">{r.score}%</span>
                              </div>
                            </div>
                          </summary>
                          <div className="req-explanation">
                            <p>{r.explanation || "No explanation available."}</p>
                            {r.resume_citation && (
                              <p className="muted small">
                                <strong>Best match:</strong> {r.resume_citation}
                              </p>
                            )}
                            {r.resume_excerpt && r.status === "missing" && (
                              <p className="muted small">
                                <strong>Closest resume text:</strong> {r.resume_excerpt}
                              </p>
                            )}
                            {!r.resume_excerpt && r.status === "missing" && (
                              <p className="muted small">
                                No similar passage was retrieved from the resume.
                              </p>
                            )}
                          </div>
                        </details>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </section>
      )}
    </div>
  );
}
