import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API = import.meta.env.VITE_API_URL || "/api";

export default function App() {
  const [session, setSession] = useState(null);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    setUploadError("");
    if (selected && !selected.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }
    setFile(selected || null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed.");

      setSession({
        id: data.session_id,
        filename: data.filename,
        pages: data.pages,
        chunks: data.chunks,
      });
      setMessages([
        {
          role: "assistant",
          content: `I've finished reading "${data.filename}" (${data.pages} page${data.pages === 1 ? "" : "s"
            }). Ask me anything about it.`,
        },
      ]);
    } catch (err) {
      setUploadError(
        err instanceof TypeError
          ? "Could not reach the server. Is the backend running?"
          : err.message
      );
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || sending || !session) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", content: question }]);
    setSending(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.id, question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to get an answer.");
      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            err instanceof TypeError
              ? "Could not reach the server. Please try again."
              : err.message,
          isError: true,
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const reset = () => {
    setSession(null);
    setFile(null);
    setMessages([]);
    setUploadError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="logo">📚</span>
          <h1>Study Chatbot</h1>
          {session && (
            <button className="new-doc-btn" onClick={reset}>
              New document
            </button>
          )}
        </div>
      </header>

      <main className="main">
        {!session ? (
          <section className="upload-panel">
            <h2>Upload a study document</h2>
            <p className="upload-hint">
              Upload a PDF and ask questions about its contents.
            </p>

            <label
              className={`dropzone ${file ? "has-file" : ""}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleFileChange({ target: { files: e.dataTransfer.files } });
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                disabled={uploading}
              />
              {file ? (
                <>
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </>
              ) : (
                <>
                  <span className="drop-icon">📄</span>
                  <span>
                    Drag &amp; drop a PDF here, or <strong>browse</strong>
                  </span>
                </>
              )}
            </label>

            {uploading && <p className="status processing">Processing document...</p>}
            {uploadError && <p className="status error">{uploadError}</p>}

            <button
              className="btn primary"
              onClick={handleUpload}
              disabled={!file || uploading}
            >
              {uploading ? "Processing..." : "Upload & Process"}
            </button>
          </section>
        ) : (
          <section className="chat-panel">
            <div className="doc-banner">
              📄 <strong>{session.filename}</strong>
              <span className="doc-meta">
                {" "}
                · {session.pages} page{session.pages === 1 ? "" : "s"} indexed
              </span>
            </div>

            <div className="chat-window">
              {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`}>
                  <div
                    className={`message ${msg.role}${msg.isError ? " error" : ""}`}
                  >
                    <div className="bubble">
                      {msg.role === "assistant" ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      ) : (
                        msg.content
                      )}

                      {msg.role === "assistant" && msg.sources?.length > 0 && (
                        <div className="sources">
                          Source:{" "}
                          {msg.sources
                            .map(
                              (s) =>
                                `${s.source}${s.page != null ? `, p. ${s.page}` : ""
                                }`
                            )
                            .join(" · ")}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {sending && (
                <div className="message-row assistant">
                  <div className="message assistant">
                    <div className="bubble typing">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form className="chat-input" onSubmit={handleSend}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about the document..."
                disabled={sending}
              />
              <button type="submit" disabled={sending || !input.trim()}>
                Send
              </button>
            </form>
          </section>
        )}
      </main>
    </div>
  );
}
