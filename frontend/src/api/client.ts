const API = import.meta.env.VITE_API_URL || "/api";

export function getToken(): string | null {
  return localStorage.getItem("token");
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem("token", token);
  else localStorage.removeItem("token");
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API}${path}`, { ...init, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  register: (email: string, password: string) =>
    request<{ access_token: string; user: { id: string; email: string } }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  login: (email: string, password: string) =>
    request<{ access_token: string; user: { id: string; email: string } }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<{ id: string; email: string }>("/auth/me"),

  /** PDF, Word (.docx), Excel (.xlsx/.xlsm), or plain .txt */
  uploadDocument: (file: File, docType = "pdf") => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("doc_type", docType);
    return request<{ document_id: string; filename: string; chunks_indexed: number }>(
      "/documents/upload",
      { method: "POST", body: fd }
    );
  },

  query: (body: {
    question: string;
    document_id?: string;
    top_k?: number;
    use_llm?: boolean;
  }) =>
    request<{
      question: string;
      answer: string | null;
      sources: { text: string; page: number; filename: string }[];
    }>("/query", { method: "POST", body: JSON.stringify(body) }),

  listDocuments: () =>
    request<
      { id: string; filename: string; doc_type: string; chunks_indexed: number; created_at: string }[]
    >("/documents"),

  deleteDocument: (documentId: string) =>
    request<{ ok: boolean }>(`/documents/${documentId}`, { method: "DELETE" }),

  queryHistory: (documentId?: string) => {
    const q = documentId ? `?document_id=${encodeURIComponent(documentId)}` : "";
    return request<
      {
        id: number;
        document_id: string | null;
        question: string;
        answer: string | null;
        source_count: number;
        created_at: string;
      }[]
    >(`/queries/history${q}`);
  },

  match: (opts: { resume: File; jdText?: string; jdFile?: File; useLlm?: boolean }) => {
    const fd = new FormData();
    fd.append("resume", opts.resume);
    if (opts.jdText?.trim()) fd.append("jd_text", opts.jdText.trim());
    if (opts.jdFile) fd.append("jd", opts.jdFile);
    fd.append("use_llm", String(opts.useLlm ?? true));
    return request<{
      overall_score: number;
      score_note: string | null;
      summary: string | null;
      strengths: string[];
      gaps: string[];
      cover_letter_topics?: string[];
      requirement_matches: {
        requirement: string;
        category: string;
        priority: string;
        status: string;
        resume_citation: string | null;
        resume_excerpt: string | null;
        explanation: string;
        score: number;
      }[];
      resume_document_id: string;
      jd_document_id: string;
    }>("/match", { method: "POST", body: fd });
  },

  summarizeMeeting: (opts: { file?: File; text?: string; useLlm?: boolean }) => {
    const fd = new FormData();
    if (opts.file) fd.append("file", opts.file);
    if (opts.text) fd.append("text", opts.text);
    fd.append("use_llm", String(opts.useLlm ?? true));
    return request<{
      summary: string;
      action_items: { task: string; owner: string | null; due: string | null }[];
    }>("/meetings/summarize", { method: "POST", body: fd });
  },
};
