import { FormEvent, useState, useRef, useEffect } from "react";
import { api } from "../api/client";
import LoadingPanel from "../components/LoadingPanel";

// TypeScript declarations for Web Speech API
interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
}

const SpeechRecognition =
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

interface ActionItem {
  id: string;
  task: string;
  owner: string | null;
  due: string | null;
  completed: boolean;
}

export default function MeetingPage() {
  // Tabs and general state
  const [activeTab, setActiveTab] = useState<"dictation" | "text">("dictation");
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");
  const [actions, setActions] = useState<ActionItem[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // Speech Recognition States
  const [isListening, setIsListening] = useState(false);
  const [language, setLanguage] = useState("ko-KR");
  const [interimTranscript, setInterimTranscript] = useState("");
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const speechSupported = !!SpeechRecognition;

  // Action Item Edit States
  const [editId, setEditId] = useState<string | null>(null);
  const [editForm, setEditForm] = useState({ task: "", owner: "", due: "" });
  
  // New Action Item States
  const [newAction, setNewAction] = useState({ task: "", owner: "", due: "" });

  // Clean up Speech Recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  // Initialize and Toggle Speech Recognition
  const toggleListening = () => {
    if (!speechSupported) {
      setError("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
      return;
    }

    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  const startListening = () => {
    setError("");
    setIsListening(true);
    setInterimTranscript("");

    try {
      const recognition = new SpeechRecognition() as SpeechRecognitionInstance;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = language;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        let interimText = "";
        let finalGenerated = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const result = event.results[i];
          if (result.isFinal) {
            finalGenerated += result[0].transcript;
          } else {
            interimText += result[0].transcript;
          }
        }

        if (finalGenerated) {
          setText((prev) => {
            const spacing = prev.trim() ? " " : "";
            return prev.trim() + spacing + finalGenerated.trim();
          });
        }
        setInterimTranscript(interimText);
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error("Speech Recognition Error:", event.error);
        if (event.error === "not-allowed") {
          setError("Microphone permission denied. Please allow microphone access in your browser settings.");
          stopListening();
        } else if (event.error !== "no-speech") {
          setError(`Speech recognition error: ${event.error}`);
          stopListening();
        }
      };

      recognition.onend = () => {
        // Double check state. Chrome sometimes terminates recognition on silence,
        // but we want continuous mode. Let's restart if state is still true
        if (recognitionRef.current === recognition && isListening) {
          try {
            recognition.start();
          } catch (e) {
            setIsListening(false);
          }
        } else {
          setIsListening(false);
        }
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      console.error(err);
      setError("Failed to start speech recognition.");
      setIsListening(false);
    }
  };

  const stopListening = () => {
    setIsListening(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setInterimTranscript("");
  };

  // Submit transcript to backend
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (isListening) stopListening();

    const isTextTab = activeTab === "text";
    if (!isTextTab && !text.trim()) {
      setError("Please dictate or type some text first.");
      return;
    }
    if (isTextTab && !file && !text.trim()) {
      setError("Please paste a transcript or upload a file first.");
      return;
    }

    setBusy(true);
    setError("");
    try {
      const res = await api.summarizeMeeting({
        file: isTextTab && file ? file : undefined,
        text: text.trim() || undefined,
        useLlm: true,
      });
      setSummary(res.summary);
      
      // Transform incoming action items to include unique IDs and completed field
      const mappedActions = res.action_items.map((item, index) => ({
        id: `item-${Date.now()}-${index}`,
        task: item.task,
        owner: item.owner,
        due: item.due,
        completed: false,
      }));
      setActions(mappedActions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Summarization failed");
    } finally {
      setBusy(false);
    }
  };

  // Action Item management
  const toggleActionCompleted = (id: string) => {
    setActions((prev) =>
      prev.map((a) => (a.id === id ? { ...a, completed: !a.completed } : a))
    );
  };

  const startEditing = (item: ActionItem) => {
    setEditId(item.id);
    setEditForm({
      task: item.task,
      owner: item.owner || "",
      due: item.due || "",
    });
  };

  const saveEdit = (id: string) => {
    if (!editForm.task.trim()) return;
    setActions((prev) =>
      prev.map((a) =>
        a.id === id
          ? {
              ...a,
              task: editForm.task.trim(),
              owner: editForm.owner.trim() || null,
              due: editForm.due.trim() || null,
            }
          : a
      )
    );
    setEditId(null);
  };

  const cancelEdit = () => {
    setEditId(null);
  };

  const deleteAction = (id: string) => {
    setActions((prev) => prev.filter((a) => a.id !== id));
  };

  const handleAddAction = (e: FormEvent) => {
    e.preventDefault();
    if (!newAction.task.trim()) return;

    const newItem: ActionItem = {
      id: `custom-${Date.now()}`,
      task: newAction.task.trim(),
      owner: newAction.owner.trim() || null,
      due: newAction.due.trim() || null,
      completed: false,
    };

    setActions((prev) => [...prev, newItem]);
    setNewAction({ task: "", owner: "", due: "" });
  };

  // Export Tools
  const copyToClipboard = () => {
    const formatted = formatReport();
    navigator.clipboard.writeText(formatted)
      .then(() => {
        alert("Meeting summary and action items copied to clipboard!");
      })
      .catch((err) => {
        setError("Failed to copy text: " + err.message);
      });
  };

  const downloadMarkdown = () => {
    const formatted = formatReport();
    const blob = new Blob([formatted], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    
    // Generate simple filename based on date
    const dateStr = new Date().toISOString().slice(0, 10);
    link.setAttribute("download", `meeting_summary_${dateStr}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatReport = () => {
    const actionList = actions.map((a) => {
      const checkbox = a.completed ? "[x]" : "[ ]";
      const ownerStr = a.owner ? ` (Owner: ${a.owner})` : "";
      const dueStr = a.due ? ` (Due: ${a.due})` : "";
      return `- ${checkbox} ${a.task}${ownerStr}${dueStr}`;
    }).join("\n");

    return `# Meeting Minutes & Action Items\n\n## Summary\n${summary}\n\n## Action Items\n${actionList || "No action items extracted."}`;
  };

  const clearAll = () => {
    if (window.confirm("Are you sure you want to clear everything?")) {
      setFile(null);
      setText("");
      setSummary("");
      setActions([]);
      setError("");
      setInterimTranscript("");
    }
  };

  return (
    <div className="stack">
      <section className="card">
        <h2>Meeting minutes summarizer (회의록 요약기)</h2>
        <p className="muted">
          Extract core action items and high-quality meeting summaries using live speech dictation or document transcripts.
        </p>

        {/* Tab Selection */}
        <div className="meeting-tabs">
          <button
            type="button"
            className={`meeting-tab ${activeTab === "dictation" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("dictation");
              setError("");
            }}
          >
            🎙️ Live Dictation (음성 인식)
          </button>
          <button
            type="button"
            className={`meeting-tab ${activeTab === "text" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("text");
              stopListening();
              setError("");
            }}
          >
            📄 Text & Document (텍스트/문서)
          </button>
        </div>

        <form onSubmit={submit} className="stack">
          {activeTab === "dictation" ? (
            <div className="dictation-container">
              <div className="dictation-controls">
                <label className="row-label">
                  <span>Language / 언어</span>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    disabled={isListening}
                  >
                    <option value="ko-KR">Korean (한국어)</option>
                    <option value="en-US">English (영어)</option>
                  </select>
                </label>

                <div
                  className={`status-badge ${isListening ? "listening" : "idle"}`}
                >
                  <span className="dot" style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    backgroundColor: isListening ? "#ef4444" : "#8b9cb3",
                    display: "inline-block",
                    animation: isListening ? "pulse-red 1s infinite alternate" : "none"
                  }} />
                  {isListening ? "Listening…" : "Ready"}
                </div>
              </div>

              <div className="mic-btn-wrapper">
                <button
                  type="button"
                  className={`microphone-btn ${isListening ? "active" : ""}`}
                  onClick={toggleListening}
                  title={isListening ? "Stop Recording" : "Start Voice Dictation"}
                >
                  {isListening ? (
                    <svg viewBox="0 0 24 24">
                      <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24">
                      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z" />
                    </svg>
                  )}
                </button>
              </div>

              {/* Animated Soundwave Visualizer */}
              <div className={`soundwave-visualizer ${isListening ? "active" : ""}`}>
                <div className="soundwave-bar"></div>
                <div className="soundwave-bar"></div>
                <div className="soundwave-bar"></div>
                <div className="soundwave-bar"></div>
                <div className="soundwave-bar"></div>
              </div>

              {interimTranscript && (
                <div className="interim-display">
                  &ldquo;{interimTranscript}&rdquo;
                </div>
              )}

              <label style={{ width: "100%", textAlign: "left" }}>
                Dictated Transcript (Edit anytime)
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={8}
                  placeholder="Your speech transcript will appear here in real-time. You can also edit it directly..."
                />
              </label>
            </div>
          ) : (
            <div className="stack">
              <label style={{ textAlign: "left" }}>
                File upload (optional)
                <span className="muted small" style={{ display: "block", marginBottom: "0.2rem" }}>
                  Upload a meeting transcript in PDF, Word (.docx), Excel (.xlsx), or TXT format.
                </span>
                <input
                  type="file"
                  accept=".pdf,.docx,.xlsx,.txt"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </label>
              
              <label style={{ textAlign: "left" }}>
                Or paste meeting notes / transcript
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={8}
                  placeholder="Paste your meeting notes, minutes, or transcript here..."
                />
              </label>
            </div>
          )}

          <div className="btn-group-row">
            <button
              type="submit"
              className="btn primary"
              disabled={busy || (!file && !text.trim())}
            >
              {busy ? "Summarizing…" : "Summarize & Extract Action Items"}
            </button>
            
            {(text || file || summary) && (
              <button
                type="button"
                className="btn ghost"
                onClick={clearAll}
              >
                Clear Everything
              </button>
            )}
          </div>

          {busy && (
            <LoadingPanel
              label="Summarizing meeting transcript"
              sublabel="Analyzing core content and extracting key action items..."
            />
          )}
        </form>
        {error && <p className="error" style={{ marginTop: "1rem" }}>{error}</p>}
      </section>

      {/* Summary Output */}
      {summary && (
        <section className="card highlight">
          <h3 style={{ textAlign: "left" }}>Meeting Summary</h3>
          <p style={{ textAlign: "left", whiteSpace: "pre-wrap", lineHeight: "1.6" }}>{summary}</p>
        </section>
      )}

      {/* Action Items Interactive Checklist */}
      {(summary || actions.length > 0) && (
        <section className="card">
          <div className="meeting-section-header">
            <h3>Key Action Items (업무 관리 체크리스트)</h3>
            <div className="btn-group-row">
              <button
                type="button"
                className="btn ghost small"
                onClick={copyToClipboard}
                title="Copy markdown report to clipboard"
              >
                📋 Copy
              </button>
              <button
                type="button"
                className="btn ghost small"
                onClick={downloadMarkdown}
                title="Download report as markdown"
              >
                💾 Download MD
              </button>
            </div>
          </div>

          {actions.length === 0 ? (
            <p className="muted" style={{ padding: "1.5rem", textAlign: "center" }}>
              No action items extracted. You can add them manually below.
            </p>
          ) : (
            <ul className="meeting-checklist">
              {actions.map((item) => (
                <li key={item.id} className="meeting-item-row">
                  {editId === item.id ? (
                    /* Inline Editing Mode Form */
                    <div className="meeting-item-edit-form">
                      <div className="edit-inputs-row">
                        <input
                          type="text"
                          name="task"
                          value={editForm.task}
                          onChange={(e) => setEditForm({ ...editForm, task: e.target.value })}
                          placeholder="Task description"
                          required
                        />
                        <input
                          type="text"
                          name="owner"
                          value={editForm.owner}
                          onChange={(e) => setEditForm({ ...editForm, owner: e.target.value })}
                          placeholder="Owner (e.g. John)"
                        />
                        <input
                          type="text"
                          name="due"
                          value={editForm.due}
                          onChange={(e) => setEditForm({ ...editForm, due: e.target.value })}
                          placeholder="Due (e.g. May 30)"
                        />
                      </div>
                      <div className="btn-group-row">
                        <button
                          type="button"
                          className="btn primary small"
                          onClick={() => saveEdit(item.id)}
                          disabled={!editForm.task.trim()}
                        >
                          Save
                        </button>
                        <button
                          type="button"
                          className="btn ghost small"
                          onClick={cancelEdit}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* Normal Display Mode */
                    <>
                      <input
                        type="checkbox"
                        className="meeting-item-checkbox"
                        checked={item.completed}
                        onChange={() => toggleActionCompleted(item.id)}
                      />
                      <div className="meeting-item-content">
                        <span
                          className={`meeting-item-text ${
                            item.completed ? "completed" : ""
                          }`}
                        >
                          {item.task}
                        </span>
                        
                        {(item.owner || item.due) && (
                          <div className="meeting-item-meta">
                            {item.owner && (
                              <span className="badge-owner">👤 {item.owner}</span>
                            )}
                            {item.due && (
                              <span className="badge-due">📅 {item.due}</span>
                            )}
                          </div>
                        )}
                      </div>

                      <div className="meeting-item-actions">
                        <button
                          type="button"
                          onClick={() => startEditing(item)}
                          title="Edit Action Item"
                        >
                          ✏️
                        </button>
                        <button
                          type="button"
                          className="delete-btn"
                          onClick={() => deleteAction(item.id)}
                          title="Delete Action Item"
                        >
                          🗑️
                        </button>
                      </div>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}

          {/* Add Custom Action Item Manual Form */}
          <form onSubmit={handleAddAction} className="add-action-form">
            <label>
              Add Custom Action
              <input
                type="text"
                placeholder="Enter custom task description…"
                value={newAction.task}
                onChange={(e) => setNewAction({ ...newAction, task: e.target.value })}
                required
              />
            </label>
            <label>
              Owner
              <input
                type="text"
                placeholder="Assignee name"
                value={newAction.owner}
                onChange={(e) => setNewAction({ ...newAction, owner: e.target.value })}
              />
            </label>
            <label>
              Due Date
              <input
                type="text"
                placeholder="Deadline"
                value={newAction.due}
                onChange={(e) => setNewAction({ ...newAction, due: e.target.value })}
              />
            </label>
            <button
              type="submit"
              className="btn primary"
              disabled={!newAction.task.trim()}
              style={{ padding: "0.45rem 1rem", height: "fit-content" }}
            >
              ➕ Add
            </button>
          </form>
        </section>
      )}
    </div>
  );
}
