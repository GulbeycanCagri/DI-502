// src/App.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// --- minimal icons (inline SVGs) ---
const Icon = {
  Sun: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M6.76 4.84l-1.8-1.79-1.41 1.41 1.79 1.8 1.42-1.42zm10.48 14.32l1.79 1.8 1.41-1.41-1.8-1.79-1.4 1.4zM12 4V1h0v3h0zm0 19v-3h0v3h0zM4 12H1v0h3v0zm19 0h-3v0h3v0zM6.76 19.16l-1.42 1.42-1.79-1.8 1.41-1.41 1.8 1.79zM18.36 4.22l1.41-1.41 1.8 1.79-1.42 1.42-1.79-1.8zM12 7a5 5 0 100 10 5 5 0 000-10z"/></svg>,
  Moon: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M21.75 15.5A9.75 9.75 0 1111.5 2.25a8 8 0 0010.25 13.25z"/></svg>,
  Send: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>,
  Mic: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M12 14a3 3 0 003-3V5a3 3 0 10-6 0v6a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 0014 0h-2zM11 19h2v3h-2z"/></svg>,
  Dollar: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M13 3h-2v2H9v2h2v10H9v2h2v2h2v-2h2v-2h-2V7h2V5h-2V3z"/></svg>,
  Globe: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1h-2v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.62-1.23 4.96-3.1 6.39z"/></svg>,
  Paperclip: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v11.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>,
  Close: (p) => <svg viewBox="0 0 24 24" width="14" height="14" {...p}><path fill="currentColor" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>,
};

const ROLE = { user: "user", assistant: "assistant", system: "system" };

/** Call backend via proxy: POST /api/chat  ->  FastAPI /chat */
async function callBackendChat(question, use_online_research, document) {
  const formData = new FormData();
  formData.append("question", question);
  formData.append("use_online_research", String(use_online_research));
  if (document) {
    formData.append("document", document);
  }

  const res = await fetch("/api/chat", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${text}`);
  }
  const data = await res.json();
  return { id: Math.random().toString(36).slice(2), role: ROLE.assistant, content: data.ai_response };
}

export default function App() {
  
  const [dark, setDark] = useState(false);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  const [messages, setMessages] = useState([]);


  // composer
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [useOnline, setUseOnline] = useState(false);
  const [doc, setDoc] = useState(null);
  const fileInputRef = useRef(null);
  const endRef = useRef(null);
  
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages.length, sending]);


  async function sendMessage(text) {
    if (!text.trim() && !doc) return;

    setSending(true);

    let userContent = text;
    if (doc) {
      userContent = `${text}\n\n[Attached: ${doc.name}]`;
    }
    const userMsg = { id: Math.random().toString(36).slice(2), role: ROLE.user, content: userContent };
    
    setMessages(prev => [...prev, userMsg]);

    try {
      const reply = await callBackendChat(text, useOnline, doc);
      setMessages(prev => [...prev, reply]);
    } catch (e) {
      const err = { id: Math.random().toString(36).slice(2), role: ROLE.assistant, content: `⚠️ Error: ${e?.message || 'Backend error'}` };
      setMessages(prev => [...prev, err]);
    } finally {
      setSending(false);
      setInput("");
      setDoc(null);
      if (fileInputRef.current) fileInputRef.current.value = null;
    }
  }

  const chips = [
    "Summarize today's market movers",
    "Show P/E and EPS for AAPL",
    "Compare BIST 100 vs S&P 500 (YTD)",
    "Which bond ETFs benefit if rates fall?",
  ];


  function prefillInput(text) {
    setInput(text);
    // İsteğe bağlı: textarea'ya odaklan
    // document.querySelector('.composer textarea')?.focus();
  }

  return (
    <div className="app">
      {/* <aside className="sidebar">
        ...
      </aside> 
      */}
      {/* --- */}

      {/* Main */}
      <section className="main">
        <div className="topbar">
          <div className="left"><strong>Finance Assistant</strong></div>
          {/* Tema değiştirme butonu sidebar'dan buraya taşındı */}
          <div className="right">
            <button className="icon" onClick={() => setDark((d) => !d)}>{dark ? <Icon.Sun /> : <Icon.Moon />}</button>
          </div>
        </div>

        <div className="messages">
          {/* Boş ekran kontrolü 'messages.length'e göre yapılıyor */}
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="logo">💰</div>
              <h1>Ask finance anything</h1>
              <p>Real-time markets, fundamentals, portfolio ideas, macro insights.</p>
              {/* Chip'ler artık 'prefillInput' fonksiyonunu çağırıyor */}
              <div className="chiprow">{chips.map((c) => <button key={c} className="chip" onClick={() => prefillInput(c)}>{c}</button>)}</div>
            </div>
          )}

          {/* Mesaj render etme işlemi 'messages' dizisine göre yapılıyor */}
          {messages.length > 0 && (
            <div className="stack">
              {messages.map((m) => (
                <div key={m.id} className={`bubble-row ${m.role === ROLE.user ? 'end' : ''}`}>
                  <div className={`bubble ${m.role === ROLE.user ? 'user' : ''}`} style={{ whiteSpace: 'pre-wrap' }}>
                    {m.content}
                  </div>
                </div>
              ))}
              {sending && <div className="typing"><span className="dot"/><span className="dot"/><span className="dot"/><span className="label">Thinking…</span></div>}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="composer">
          {doc && (
            <div className="file-preview">
              <span>{doc.name}</span>
              <button className="icon" title="Remove file" onClick={() => { setDoc(null); if (fileInputRef.current) fileInputRef.current.value = null; }}>
                <Icon.Close />
              </button>
            </div>
          )}
          <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about stocks, ETFs, bonds, crypto, macro…" />
          <div className="send">
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              onChange={(e) => setDoc(e.target.files[0] || null)} 
            />
            <button className="icon" title="Attach document" onClick={() => fileInputRef.current?.click()}>
              <Icon.Paperclip />
            </button>
            <button 
              className={`icon ${useOnline ? "active" : ""}`} 
              title={`Online research: ${useOnline ? 'ON' : 'OFF'}`}
              onClick={() => setUseOnline(v => !v)}
            >
              <Icon.Globe />
            </button>
            <button className="icon" title="Voice"><Icon.Mic /></button>
            <button className="btn primary" disabled={sending || (!input.trim() && !doc)} onClick={() => sendMessage(input)}>
              <Icon.Send /> Send
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}