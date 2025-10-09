// src/App.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// --- minimal icons (inline SVGs) ---
const Icon = {
  Sun: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M6.76 4.84l-1.8-1.79-1.41 1.41 1.79 1.8 1.42-1.42zm10.48 14.32l1.79 1.8 1.41-1.41-1.8-1.79-1.4 1.4zM12 4V1h0v3h0zm0 19v-3h0v3h0zM4 12H1v0h3v0zm19 0h-3v0h3v0zM6.76 19.16l-1.42 1.42-1.79-1.8 1.41-1.41 1.8 1.79zM18.36 4.22l1.41-1.41 1.8 1.79-1.42 1.42-1.79-1.8zM12 7a5 5 0 100 10 5 5 0 000-10z"/></svg>,
  Moon: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M21.75 15.5A9.75 9.75 0 1111.5 2.25a8 8 0 0010.25 13.25z"/></svg>,
  Plus: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M11 11V5h2v6h6v2h-6v6h-2v-6H5v-2z"/></svg>,
  Trash: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 7h2v8h-2v-8zm4 0h2v8h-2v-8zM6 10h2v8H6v-8z"/></svg>,
  Send: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>,
  Mic: (p) => <svg viewBox="0 0 24 24" width="18" height="18" {...p}><path fill="currentColor" d="M12 14a3 3 0 003-3V5a3 3 0 10-6 0v6a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 0014 0h-2zM11 19h2v3h-2z"/></svg>,
  Search: (p) => <svg viewBox="0 0 24 24" width="16" height="16" {...p}><path fill="currentColor" d="M10 18a8 8 0 105.3-14.3A8 8 0 0010 18zm11 3l-4.3-4.3" stroke="currentColor" strokeWidth="2"/></svg>,
  Dollar: (p) => <svg viewBox="0 0 24 24" width="20" height="20" {...p}><path fill="currentColor" d="M13 3h-2v2H9v2h2v10H9v2h2v2h2v-2h2v-2h-2V7h2V5h-2V3z"/></svg>,
};

const ROLE = { user: "user", assistant: "assistant", system: "system" };

// Mock backend (replace with real fetch)
async function fakeChat(messages) {
  await new Promise((r) => setTimeout(r, 500));
  const last = messages[messages.length - 1]?.content || "";
  return { id: Math.random().toString(36).slice(2), role: ROLE.assistant, content: `**Summary:** ${last}\nData sources: Alpha Vantage, Yahoo Finance.` };
}

export default function App() {
  // theme
  const [dark, setDark] = useState(false);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  // conversations
  const [convs, setConvs] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const active = useMemo(() => convs.find((c) => c.id === activeId) || null, [convs, activeId]);

  // composer
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const endRef = useRef(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [active?.messages?.length, sending]);

  function newChat(prefill = "") {
    const id = Math.random().toString(36).slice(2);
    const msgs = prefill ? [{ id: Date.now(), role: ROLE.user, content: prefill }] : [];
    const conv = { id, title: prefill || "New chat", createdAt: Date.now(), messages: msgs };
    setConvs((p) => [conv, ...p]);
    setActiveId(id);
  }

  function deleteChat(id) {
    setConvs((p) => p.filter((c) => c.id !== id));
    if (id === activeId) setActiveId((prev) => (convs.find((c) => c.id !== id)?.id || null));
  }

  async function sendMessage(text) {
    if (!text.trim()) return;

    // Ensure there is an active chat
    let id = activeId;
    if (!id) {
      const nid = Math.random().toString(36).slice(2);
      const conv = { id: nid, title: text.slice(0, 36) || "New chat", createdAt: Date.now(), messages: [] };
      setConvs((p) => [conv, ...p]);
      setActiveId(nid);
      id = nid;
    }

    setSending(true);

    // append user message
    setConvs((p) => p.map((c) => c.id === id ? { ...c, title: c.title === "New chat" ? text.slice(0,36) : c.title, messages: [...c.messages, { id: Math.random().toString(36).slice(2), role: ROLE.user, content: text }] } : c));

    try {
      // build message list for mock backend
      const curr = convs.find((c) => c.id === id) || { messages: [] };
      const msgs = [...curr.messages, { role: ROLE.user, content: text }].map(({ role, content }) => ({ role, content }));
      const res = await fakeChat(msgs);
      setConvs((p) => p.map((c) => c.id === id ? { ...c, messages: [...c.messages, res] } : c));
    } catch (e) {
      setConvs((p) => p.map((c) => c.id === id ? { ...c, messages: [...c.messages, { id: Math.random().toString(36).slice(2), role: ROLE.assistant, content: `⚠️ Error: ${e?.message || 'Backend error'}` }] } : c));
    } finally {
      setSending(false);
      setInput("");
    }
  }

  const chips = [
    "Summarize today's market movers",
    "Show P/E and EPS for AAPL",
    "Compare BIST 100 vs S&P 500 (YTD)",
    "Which bond ETFs benefit if rates fall?",
  ];

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="side-top">
          <div className="brand"><Icon.Dollar /> <span>Finance Chat</span></div>
          <button className="icon" onClick={() => setDark((d) => !d)}>{dark ? <Icon.Sun /> : <Icon.Moon />}</button>
        </div>
        <div className="side-controls">
          <button className="btn primary" onClick={() => newChat("")}> <Icon.Plus /> New chat</button>
          <div className="search"><Icon.Search /><input placeholder="Search chats" /></div>
        </div>
        <div className="conv-list">
          {convs.length === 0 && <div className="empty">No conversations yet.</div>}
          {convs.map((c) => (
            <div key={c.id} className={`conv ${c.id === activeId ? "active" : ""}`} onClick={() => setActiveId(c.id)}>
              <div className="title">{c.title}</div>
              <div className="meta">{new Date(c.createdAt).toLocaleDateString()} <button className="icon" onClick={(e)=>{e.stopPropagation(); deleteChat(c.id);}}><Icon.Trash/></button></div>
            </div>
          ))}
        </div>
      </aside>

      {/* Main */}
      <section className="main">
        <div className="topbar">
          <div className="left"><strong>Finance Assistant</strong></div>
          <div className="right"><button className="btn">Finance‑GPT ▾</button></div>
        </div>

        <div className="messages">
          {!active && (
            <div className="empty-state">
              <div className="logo">💰</div>
              <h1>Ask finance anything</h1>
              <p>Real‑time markets, fundamentals, portfolio ideas, macro insights.</p>
              <div className="chiprow">{chips.map((c) => <button key={c} className="chip" onClick={() => newChat(c)}>{c}</button>)}</div>
            </div>
          )}

          {active && (
            <div className="stack">
              {active.messages.map((m) => (
                <div key={m.id} className={`bubble-row ${m.role === ROLE.user ? 'end' : ''}`}>
                  <div className={`bubble ${m.role === ROLE.user ? 'user' : ''}`}>{m.content}</div>
                </div>
              ))}
              {sending && <div className="typing"><span className="dot"/><span className="dot"/><span className="dot"/><span className="label">Thinking…</span></div>}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="composer">
          <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about stocks, ETFs, bonds, crypto, macro…" />
          <div className="send">
            <button className="icon" title="Voice"><Icon.Mic /></button>
            <button className="btn primary" disabled={sending || !input.trim()} onClick={() => sendMessage(input)}><Icon.Send /> Send</button>
          </div>
        </div>
      </section>
    </div>
  );
}
