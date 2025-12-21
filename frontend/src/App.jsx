import React, { useEffect, useRef, useState } from "react";
import "./App.css";

// --- Minimal Icons ---
const Icon = {
  Sun: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M6.76 4.84l-1.8-1.79-1.41 1.41 1.79 1.8 1.42-1.42zm10.48 14.32l1.79 1.8 1.41-1.41-1.8-1.79-1.4 1.4zM12 4V1h0v3h0zm0 19v-3h0v3h0zM4 12H1v0h3v0zm19 0h-3v0h3v0zM6.76 19.16l-1.42 1.42-1.79-1.8 1.41-1.41 1.8 1.79zM18.36 4.22l1.41-1.41 1.8 1.79-1.42 1.42-1.79-1.8zM12 7a5 5 0 100 10 5 5 0 000-10z"/></svg>,
  Moon: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M21.75 15.5A9.75 9.75 0 1111.5 2.25a8 8 0 0010.25 13.25z"/></svg>,
  Send: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>,
  Stop: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><rect x="6" y="6" width="12" height="12" fill="currentColor" /></svg>,
  Globe: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1h-2v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.62-1.23 4.96-3.1 6.39z"/></svg>,
  Paperclip: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v11.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>,
  Close: (p) => <svg viewBox="0 0 24 24" width="14" height="14" {...p}><path fill="currentColor" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 17.59 13.41 12z"/></svg>,
  NewChat: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>,
};

const ROLE = { user: "user", assistant: "assistant" };
const API_URL = "/api/chat"; 
//const API_URL = "https://34.139.154.101.nip.io/chat";

// Session storage key
const SESSION_STORAGE_KEY = "economind_session_id";

// Generate a UUID for new sessions
function generateSessionId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Get or create session ID from localStorage
function getOrCreateSessionId() {
  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }
  return sessionId;
}

async function streamBackendChat(question, use_online_research, document, sessionId, signal, onChunk, onSessionId, onDone, onError) {
  const formData = new FormData();
  formData.append("question", question);
  formData.append("use_online_research", String(use_online_research));
  formData.append("session_id", sessionId);
  if (document) {
    formData.append("document", document);
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
      signal: signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "Unknown Error");
      throw new Error(`Server Error (${response.status}): ${text}`);
    }

    if (!response.body) throw new Error("Response body is empty.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let isFirstChunk = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      let chunk = decoder.decode(value, { stream: true });
      
      // Handle session ID from first chunk
      if (isFirstChunk) {
        const sessionMatch = chunk.match(/\[SESSION_ID:([^\]]+)\]/);
        if (sessionMatch) {
          const newSessionId = sessionMatch[1];
          onSessionId(newSessionId);
          chunk = chunk.replace(/\[SESSION_ID:[^\]]+\]/, '');
        }
        isFirstChunk = false;
      }
      
      if (chunk) {
        onChunk(chunk);
      }
    }
    
    onDone();

  } catch (err) {
    if (err.name === 'AbortError') {
      console.log("User stopped the generation.");
      onDone();
      return;
    }

    let userMessage = err.message;
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      userMessage = "Connection Error: Cannot reach the backend.";
    }
    console.error("Stream Error:", err);
    onError(new Error(userMessage));
  }
}

export default function App() {
  const [dark, setDark] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [useOnline, setUseOnline] = useState(false);
  const [doc, setDoc] = useState(null);
  const [sessionId, setSessionId] = useState(() => getOrCreateSessionId());

  const fileInputRef = useRef(null);
  const endRef = useRef(null);
  const abortControllerRef = useRef(null);

  useEffect(() => { 
    document.documentElement.dataset.theme = dark ? "dark" : "light"; 
  }, [dark]);

  useEffect(() => { 
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  // Start a new conversation (clears messages and creates new session)
  function startNewChat() {
    if (sending) return; // Don't allow during generation
    
    // Generate new session ID
    const newSessionId = generateSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
    setSessionId(newSessionId);
    
    // Clear messages
    setMessages([]);
    setInput("");
    setDoc(null);
    if (fileInputRef.current) fileInputRef.current.value = null;
    
    console.log("Started new chat session:", newSessionId);
  }

  function stopGenerating() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      

      setMessages(prev => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        

        if (lastMsg && lastMsg.role === ROLE.assistant) {
            const suffix = lastMsg.content ? " ... [Stopped]" : "[Generation Cancelled]";
            
            return updated.map((msg, idx) => 
                idx === updated.length - 1 
                ? { ...msg, content: msg.content + suffix } 
                : msg
            );
        }
        return updated;
      });

      setSending(false);
    }
  }

  async function sendMessage(text) {
    if (!text.trim() && !doc) return;

    setSending(true);
    
    abortControllerRef.current = new AbortController();

    let userContent = text;
    if (doc) {
      userContent = `${text}\n\n[Attached: ${doc.name}]`;
    }
    const userMsg = { id: Math.random().toString(36).slice(2), role: ROLE.user, content: userContent };
    const assistantId = Math.random().toString(36).slice(2);
    const assistantMsg = { id: assistantId, role: ROLE.assistant, content: "" };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    
    setInput("");
    setDoc(null);
    if (fileInputRef.current) fileInputRef.current.value = null;

    await streamBackendChat(
      text,
      useOnline,
      doc,
      sessionId,
      abortControllerRef.current.signal,
      (chunk) => {
        setMessages(prev => prev.map(msg => 
          msg.id === assistantId ? { ...msg, content: msg.content + chunk } : msg
        ));
      },
      (newSessionId) => {
        // Update session ID if backend provides a new one
        if (newSessionId && newSessionId !== sessionId) {
          localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
          setSessionId(newSessionId);
        }
      },
      () => {
        setSending(false);
        abortControllerRef.current = null;
      },
      (err) => {
        setMessages(prev => prev.map(msg => 
          msg.id === assistantId ? { ...msg, content: msg.content + `\n\n[⚠️ Error: ${err.message}]` } : msg
        ));
        setSending(false);
        abortControllerRef.current = null;
      }
    );
  }

  const chips = [
    "Summarize today's market movers",
    "Summarize news for AAPL",
    "Which bond ETFs benefit if rates fall?",
  ];

  return (
    <div className="app">
      <section className="main">
        <div className="topbar">
          <div className="left">
            <button 
              className="icon new-chat-btn" 
              onClick={startNewChat} 
              disabled={sending}
              title="Start New Chat"
            >
              <Icon.NewChat />
            </button>
            <strong>Finance Assistant</strong>
          </div>
          <div className="right">
            <button className="icon" onClick={() => setDark((d) => !d)}>
              {dark ? <Icon.Sun /> : <Icon.Moon />}
            </button>
          </div>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="logo">💰</div>
              <h1>Ask finance anything</h1>
              <p>Real-time markets, fundamentals, portfolio ideas, macro insights.</p>
              <div className="chiprow">
                {chips.map((c) => (
                  <button key={c} className="chip" onClick={() => setInput(c)}>{c}</button>
                ))}
              </div>
            </div>
          )}

          {messages.length > 0 && (
            <div className="stack">
              {messages.map((m, index) => {
                const isLastMessage = index === messages.length - 1;
                const isStreaming = m.role === ROLE.assistant && sending && isLastMessage;

                return (
                  <div key={m.id} className={`bubble-row ${m.role === ROLE.user ? 'end' : ''}`}>
                    <div className={`bubble ${m.role === ROLE.user ? 'user' : ''}`}>
                      
                      {m.role === ROLE.assistant && !m.content && isStreaming && (
                        <span className="loading-text">Answer is being generated...</span>
                      )}

                      {m.content}

                      {m.content && isStreaming && <span className="cursor"></span>}

                    </div>
                  </div>
                );
              })}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="composer">
          {doc && (
            <div className="file-preview">
              <span>{doc.name}</span>
              <button 
                className="icon" 
                title="Remove file" 
                onClick={() => { setDoc(null); if (fileInputRef.current) fileInputRef.current.value = null; }}
              >
                <Icon.Close />
              </button>
            </div>
          )}
          
          <textarea 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Ask about stocks, ETFs, bonds, crypto, macro…"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(input);
              }
            }}
          />
          
          <div className="send">
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) {
                  setDoc(file);
                  setUseOnline(false); 
                }
              }}
            />
            
            <button 
              className={`icon ${doc ? "active" : ""}`} 
              title="Attach document" 
              onClick={() => fileInputRef.current?.click()}
            >
              <Icon.Paperclip />
            </button>
            
            <button
              className={`icon ${useOnline ? "active" : ""}`}
              title={`Online research: ${useOnline ? 'ON' : 'OFF'}`}
              onClick={() => {
                if (!useOnline) {
                  setDoc(null); 
                  if (fileInputRef.current) fileInputRef.current.value = null;
                }
                setUseOnline(v => !v);
              }}
            >
              <Icon.Globe />
            </button>
            
            {/**/}
            {sending ? (
                <button className="btn danger" onClick={stopGenerating}>
                    <Icon.Stop /> Stop
                </button>
            ) : (
                <button className="btn primary" disabled={!input.trim() && !doc} onClick={() => sendMessage(input)}>
                    <Icon.Send /> Send
                </button>
            )}

          </div>
        </div>
      </section>
    </div>
  );
}