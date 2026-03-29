const { useEffect, useMemo, useRef, useState } = React;

const API_BASE = "/api";
const CSS = `
  * { box-sizing: border-box; }
  body { color: #e6eefc; font-family: "IBM Plex Sans","Segoe UI",sans-serif; }
  .ra-app { display: flex; height: 100vh; background: radial-gradient(circle at top right, rgba(122,162,255,.18), transparent 30%), linear-gradient(180deg, #09101d 0%, #0e1728 100%); }
  .ra-side { width: 268px; display: flex; flex-direction: column; gap: 16px; padding: 22px 18px; border-right: 1px solid rgba(166,186,228,.15); background: linear-gradient(180deg, rgba(14,23,40,.96), rgba(10,17,30,.98)); }
  .ra-brand { display: flex; flex-direction: column; gap: 6px; }
  .ra-brand-row { display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: .04em; }
  .ra-brand-badge { width: 30px; height: 30px; display: grid; place-items: center; border-radius: 10px; color: #09101d; background: linear-gradient(135deg, rgba(209,233,255,.98), rgba(136,181,255,.95)); box-shadow: 0 12px 24px rgba(45,80,145,.32); }
  .ra-subtitle, .ra-section { color: #7f90b2; font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
  .ra-subtitle { font-size: 12px; }
  .ra-btn, .ra-chip, .ra-mode, .ra-send, .ra-select, .ra-input, .ra-top, .ra-sub, .ra-panel { border: 1px solid rgba(145,164,203,.14); }
  .ra-btn { background: rgba(20,30,51,.92); color: #dbe7ff; padding: 12px 14px; border-radius: 16px; text-align: left; cursor: pointer; }
  .ra-list { display: flex; flex-direction: column; gap: 10px; min-height: 0; overflow: auto; }
  .ra-chip { display: flex; align-items: center; gap: 10px; min-height: 48px; padding: 10px 12px; border-radius: 14px; background: rgba(18,28,48,.82); color: #afbdd7; }
  .ra-chip-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ra-dot { width: 9px; height: 9px; border-radius: 999px; background: linear-gradient(135deg, #79a3ff, #9de3ff); box-shadow: 0 0 0 4px rgba(121,163,255,.12); }
  .ra-x { border: none; background: transparent; color: #8ea3c8; cursor: pointer; }
  .ra-modes { display: flex; flex-direction: column; gap: 10px; }
  .ra-mode { display: flex; gap: 12px; padding: 14px; border-radius: 18px; background: rgba(17,25,43,.84); cursor: pointer; color: inherit; text-align: left; }
  .ra-mode-icon { width: 34px; height: 34px; border-radius: 12px; display: grid; place-items: center; font-size: 14px; font-weight: 700; color: #06101f; flex-shrink: 0; }
  .ra-mode-title { font-size: 14px; font-weight: 700; color: #edf4ff; }
  .ra-mode-desc { font-size: 12px; line-height: 1.4; color: #91a2c5; }
  .ra-profile { margin-top: auto; padding: 12px 14px; border-radius: 16px; background: rgba(14,33,28,.95); border: 1px solid rgba(52,211,153,.2); color: #9ee3c8; font-size: 13px; line-height: 1.5; }
  .ra-main { flex: 1; display: flex; flex-direction: column; min-width: 0; padding: 20px; gap: 16px; }
  .ra-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 20px; border-radius: 22px; background: rgba(13,22,38,.9); box-shadow: 0 18px 40px rgba(0,0,0,.22); }
  .ra-top-left { display: flex; align-items: center; gap: 14px; }
  .ra-title { font-size: 20px; font-weight: 700; color: #f8fbff; }
  .ra-meta { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; color: #8ea3c8; font-size: 13px; }
  .ra-badge { padding: 6px 10px; border-radius: 999px; background: rgba(255,255,255,.04); border: 1px solid rgba(166,186,228,.12); }
  .ra-sub { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 18px; border-radius: 18px; background: rgba(12,19,34,.86); color: #9eb0d4; }
  .ra-select { min-width: 260px; padding: 10px 12px; border-radius: 12px; background: rgba(19,29,48,.95); color: #edf4ff; }
  .ra-panel { position: relative; flex: 1; min-height: 0; overflow: hidden; border-radius: 28px; background: linear-gradient(180deg, rgba(11,18,32,.9), rgba(12,20,35,.96)); box-shadow: 0 24px 60px rgba(0,0,0,.24); }
  .ra-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(126,159,219,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(126,159,219,.05) 1px, transparent 1px); background-size: 26px 26px; mask-image: linear-gradient(180deg, rgba(0,0,0,.85), rgba(0,0,0,.2)); }
  .ra-scroll { position: absolute; inset: 0; overflow: auto; padding: 22px; }
  .ra-messages { display: flex; flex-direction: column; gap: 18px; min-height: 100%; }
  .ra-empty { min-height: 100%; display: grid; place-items: center; text-align: center; padding: 24px; }
  .ra-empty-card { max-width: 540px; padding: 28px 30px; border-radius: 24px; background: rgba(15,24,43,.86); border: 1px solid rgba(149,170,210,.14); }
  .ra-empty-title { font-size: 24px; font-weight: 700; color: #f7fbff; margin-bottom: 10px; }
  .ra-empty-text { font-size: 15px; line-height: 1.7; color: #94a7ca; }
  .ra-divider { display: flex; align-items: center; gap: 12px; }
  .ra-line { flex: 1; height: 1px; background: rgba(142,163,200,.16); }
  .ra-divider-badge { display: flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; background: rgba(16,24,40,.96); border: 1px solid rgba(145,164,203,.16); font-size: 11px; color: #9bb0d4; text-transform: uppercase; letter-spacing: .08em; }
  .ra-user, .ra-assistant { max-width: 85%; }
  .ra-user { align-self: flex-end; }
  .ra-assistant { align-self: flex-start; }
  .ra-tag { margin-bottom: 6px; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
  .ra-user-bubble { padding: 12px 16px; border-radius: 18px 18px 6px 18px; color: white; white-space: pre-wrap; line-height: 1.55; }
  .ra-assistant-bubble { padding: 16px; border-radius: 18px 18px 18px 6px; background: #181c2e; color: #f3f8ff; white-space: pre-wrap; line-height: 1.6; }
  .ra-cites { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(145,164,203,.14); }
  .ra-cites-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .ra-cites-title { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #9cb3da; font-weight: 700; }
  .ra-cites-count { min-width: 22px; height: 22px; padding: 0 8px; border-radius: 999px; background: rgba(96,165,250,.14); border: 1px solid rgba(96,165,250,.34); color: #b9d4ff; font-size: 11px; display: grid; place-items: center; font-weight: 700; }
  .ra-cite { padding: 12px 14px; border-radius: 12px; background: rgba(10,16,30,.88); border: 1px solid rgba(145,164,203,.16); }
  .ra-cite-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
  .ra-cite-label { font-size: 12px; font-weight: 700; color: #e3eeff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ra-cite-badge { padding: 2px 7px; border-radius: 999px; background: rgba(96,165,250,.15); border: 1px solid rgba(96,165,250,.35); font-size: 10px; letter-spacing: .06em; color: #9ec6ff; font-weight: 700; }
  .ra-cite-meta { font-size: 11px; color: #95a9ce; }
  .ra-cite-text { font-size: 12px; line-height: 1.55; color: #a9bddf; }
  .ra-cite-block { display: flex; flex-direction: column; gap: 6px; }
  .ra-error { padding: 12px 14px; border-radius: 14px; background: rgba(61,20,28,.9); border: 1px solid rgba(251,146,60,.35); color: #ffd0ad; font-size: 13px; }
  .ra-dots { display: flex; gap: 8px; padding-left: 24px; }
  .ra-pulse { width: 8px; height: 8px; border-radius: 999px; animation: raPulse 1.2s infinite ease-in-out; }
  .ra-inputbar { display: flex; align-items: flex-end; gap: 14px; padding: 16px 18px; border-radius: 24px; background: rgba(12,19,34,.92); border: 1px solid rgba(145,164,203,.12); }
  .ra-input { flex: 1; min-height: 56px; max-height: 120px; resize: none; padding: 16px 18px; border-radius: 18px; background: rgba(19,29,48,.95); color: #edf5ff; font: inherit; line-height: 1.5; }
  .ra-send { min-width: 120px; height: 56px; border-radius: 18px; color: #06101f; font-weight: 700; cursor: pointer; }
  @keyframes raPulse { 0%,80%,100% { transform: scale(.75); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
`;

const MODES = [
  { id: "local", name: "Local Brain", glyph: "L", hex: "#60a5fa", desc: "Answers only from your papers" },
  { id: "global", name: "Global Brain", glyph: "G", hex: "#a78bfa", desc: "Full reasoning plus paper context" },
  { id: "writer", name: "Paper Writer", glyph: "W", hex: "#34d399", desc: "Drafts in your research voice" },
  { id: "reviewer", name: "Reviewer", glyph: "R", hex: "#fb923c", desc: "Live skeptic vs advocate debate" },
  { id: "comparator", name: "Comparator", glyph: "C", hex: "#f472b6", desc: "Compare up to three papers" },
];
const REVIEW_LENSES = [
  { id: "full", label: "Full Review" },
  { id: "novelty", label: "Novelty" },
  { id: "method", label: "Methods" },
];

function modeOf(id) { return MODES.find((mode) => mode.id === id) || MODES[0]; }
function sessionId() { return (typeof crypto !== "undefined" && crypto.randomUUID) ? crypto.randomUUID() : `session-${Date.now()}`; }
function clip(text, limit) { return text && text.length > limit ? `${text.slice(0, limit)}...` : (text || ""); }
function reviewLensLabel(id) { return (REVIEW_LENSES.find((lens) => lens.id === id) || REVIEW_LENSES[0]).label; }
function buildReviewerPrompt(userPrompt, lensId) {
  const lens = reviewLensLabel(lensId);
  const trimmed = (userPrompt || "").trim();
  if (!trimmed) {
    return `[Start Debate] Focus lens: ${lens}`;
  }
  return trimmed;
}
function normalizeSnippet(text) { return (text || "").replace(/\s+/g, " ").trim(); }
function normalizeCitations(citations) {
  const seen = new Set();
  const cleaned = [];
  for (const citation of citations || []) {
    const key = [
      citation.paper_id || "",
      citation.chunk_id || "",
      citation.page || "",
      citation.filename || "",
    ].join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    cleaned.push({ ...citation, snippet: normalizeSnippet(citation.snippet || "") });
  }
  return cleaned;
}

function simplifyError(detail) {
  const text = String(detail || "").trim();
  if (/rate limit reached/i.test(text) && /groq|tokens per day|rate_limit_exceeded/i.test(text)) {
    return "Groq quota reached. Auto-failover is active, so responses may be slightly slower for now.";
  }
  return text || "Request failed.";
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (error) {}
    throw new Error(simplifyError(detail));
  }
  return response.json();
}

function ResearchAgent() {
  const [sid] = useState(() => sessionId());
  const [activeMode, setActiveMode] = useState("global");
  const [draft, setDraft] = useState("");
  const [papers, setPapers] = useState([]);
  const [selectedPaperIds, setSelectedPaperIds] = useState([]);
  const [reviewPaperId, setReviewPaperId] = useState("");
  const [reviewLens, setReviewLens] = useState("full");
  const reviewInterventionMode = "ask";
  const [history, setHistory] = useState([]);
  const [styleProfile, setStyleProfile] = useState({ active: false, profile: "", source_count: 0 });
  const [health, setHealth] = useState({ llm_available: false, indexed_papers: 0 });
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);
  const scrollRef = useRef(null);
  const didInitialResetRef = useRef(false);
  const currentMode = useMemo(() => modeOf(activeMode), [activeMode]);
  const reviewerDebug = useMemo(() => {
    for (let i = history.length - 1; i >= 0; i -= 1) {
      const item = history[i];
      if (item.role === "assistant" && item.mode === "reviewer" && item.debug) return item.debug;
    }
    return {};
  }, [history]);

  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  useEffect(() => { bootstrap({ resetOnLoad: true }); }, []);
  useEffect(() => {
    if (!papers.length) {
      setReviewPaperId("");
      setSelectedPaperIds([]);
      return;
    }
    if (!reviewPaperId || !papers.some((paper) => paper.paper_id === reviewPaperId)) {
      setReviewPaperId(papers[0].paper_id);
    }
    setSelectedPaperIds((current) => current.filter((id) => papers.some((paper) => paper.paper_id === id)));
  }, [papers]);
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, loading, error]);

  async function bootstrap({ resetOnLoad = false } = {}) {
    try {
      if (resetOnLoad && !didInitialResetRef.current) {
        didInitialResetRef.current = true;
        try {
          await api(`${API_BASE}/papers`, { method: "DELETE" });
        } catch (clearErr) {}
      }
      const [paperData, profileData, healthData] = await Promise.all([
        api(`${API_BASE}/papers`),
        api(`${API_BASE}/style-profile`),
        api("/health"),
      ]);
      setPapers(paperData.papers || []);
      setStyleProfile(profileData);
      setHealth(healthData);
    } catch (err) {
      setError(err.message || "Could not load app state.");
    }
  }

  function switchMode(modeId) {
    if (modeId === activeMode) return;
    setActiveMode(modeId);
    if (modeId === "comparator" && selectedPaperIds.length < 2 && papers.length >= 2) {
      setSelectedPaperIds(papers.slice(0, 2).map((paper) => paper.paper_id));
    }
    setHistory((current) => current.concat({ type: "divider", id: `${Date.now()}-${modeId}`, mode: modeId }));
  }

  function toggleComparator(paperId) {
    setSelectedPaperIds((current) => {
      if (current.includes(paperId)) return current.filter((id) => id !== paperId);
      if (current.length >= 3) return current;
      return current.concat(paperId);
    });
  }

  async function uploadFiles(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    setUploading(true);
    setError("");
    try {
      await api(`${API_BASE}/papers/upload`, { method: "POST", body: form });
      await bootstrap();
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function removePaper(paperId) {
    setError("");
    try {
      await api(`${API_BASE}/papers/${paperId}`, { method: "DELETE" });
      await bootstrap();
    } catch (err) {
      setError(err.message || "Delete failed.");
    }
  }

  function compactHistory() {
    return history
      .filter((item) => item.role === "user" || item.role === "assistant")
      .slice(-8)
      .map((item) => ({ role: item.role, content: item.content }));
  }

  async function send() {
    const rawMessage = draft.trim();
    if (loading) return;
    if (activeMode === "reviewer" && !reviewPaperId) {
      setError("Reviewer mode needs a selected paper.");
      return;
    }
    if (activeMode === "comparator" && selectedPaperIds.length < 2) {
      setError("Comparator mode needs at least two selected papers.");
      return;
    }

    const message = activeMode === "reviewer" ? buildReviewerPrompt(rawMessage, reviewLens) : rawMessage;
    if (!message) return;
    const userVisibleMessage = activeMode === "reviewer"
      ? (rawMessage || `[Start Debate] ${reviewLensLabel(reviewLens)}`)
      : rawMessage;

    const historyPayload = compactHistory();
    setDraft("");
    setError("");
    setLoading(true);
    setHistory((current) => current.concat({ id: `${Date.now()}-u`, role: "user", mode: activeMode, content: userVisibleMessage }));
    try {
      const response = await api(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sid,
          mode: activeMode,
          message,
          paper_ids: activeMode === "comparator" ? selectedPaperIds : [],
          review_paper_id: activeMode === "reviewer" ? reviewPaperId : null,
          intervention_mode: activeMode === "reviewer" ? reviewInterventionMode : null,
          history: historyPayload,
        }),
      });
      setHistory((current) => current.concat({
        id: `${Date.now()}-a`,
        role: "assistant",
        mode: activeMode,
        content: response.answer,
        citations: normalizeCitations(response.citations || []),
        debug: response.debug || {},
      }));
    } catch (err) {
      setError(err.message || "Chat failed.");
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className="ra-app">
      <aside className="ra-side">
        <div className="ra-brand">
          <div className="ra-brand-row"><div className="ra-brand-badge">RA</div><div>Research Agent</div></div>
          <div className="ra-subtitle">Pinecone + Gemini + Groq</div>
        </div>
        <input ref={fileRef} type="file" accept=".pdf,application/pdf" multiple style={{ display: "none" }} onChange={uploadFiles} />
        <button className="ra-btn" disabled={uploading} onClick={() => fileRef.current && fileRef.current.click()}>
          {uploading ? "Uploading and indexing..." : "Upload PDF"}
          <div style={{ marginTop: 4, fontSize: 12, color: "#8ea3c8" }}>Hybrid retrieval: dense + sparse + rerank</div>
        </button>

        <div className="ra-section">Paper Library</div>
        <div className="ra-list">
          {!papers.length ? <div className="ra-chip"><div className="ra-dot" /><div className="ra-chip-name">No papers uploaded yet</div></div> : papers.map((paper) => {
            const selected = selectedPaperIds.includes(paper.paper_id);
            const comparator = activeMode === "comparator";
            const disabled = comparator && !selected && selectedPaperIds.length >= 3;
            return (
              <div
                key={paper.paper_id}
                className="ra-chip"
                style={{ borderColor: selected ? `${modeOf("comparator").hex}66` : undefined, cursor: comparator ? "pointer" : "default" }}
                onClick={() => comparator && !disabled && toggleComparator(paper.paper_id)}
              >
                {comparator ? <input type="checkbox" checked={selected} disabled={disabled} onClick={(event) => event.stopPropagation()} onChange={() => toggleComparator(paper.paper_id)} /> : null}
                <div className="ra-dot" />
                <div className="ra-chip-name" title={paper.filename}>{paper.filename}</div>
                <button className="ra-x" type="button" onClick={(event) => { event.stopPropagation(); removePaper(paper.paper_id); }}>x</button>
              </div>
            );
          })}
        </div>

        <div className="ra-section">Modes</div>
        <div className="ra-modes">
          {MODES.map((mode) => {
            const active = mode.id === activeMode;
            return (
              <button
                key={mode.id}
                type="button"
                className="ra-mode"
                onClick={() => switchMode(mode.id)}
                style={{ borderColor: active ? `${mode.hex}66` : undefined, background: active ? `linear-gradient(135deg, ${mode.hex}20, rgba(17,25,43,.96))` : undefined }}
              >
                <div className="ra-mode-icon" style={{ background: `linear-gradient(135deg, ${mode.hex}, #f8fbff)` }}>{mode.glyph}</div>
                <div>
                  <div className="ra-mode-title">{mode.name}</div>
                  <div className="ra-mode-desc">{mode.desc}</div>
                </div>
              </button>
            );
          })}
        </div>

        <div className="ra-profile">
          {styleProfile.active ? "Style profile active" : "Style profile inactive"}
          <div style={{ marginTop: 6, color: "#75c7a6", fontSize: 12 }}>
            {styleProfile.active ? `${styleProfile.source_count} paper(s) in writer memory. ${clip(styleProfile.profile, 120)}` : "Upload papers to build a persistent writing profile."}
          </div>
        </div>
      </aside>

      <main className="ra-main">
        <div className="ra-top">
          <div className="ra-top-left">
            <div className="ra-brand-badge" style={{ width: 52, height: 52, borderRadius: 18, fontSize: 16, background: `linear-gradient(135deg, ${currentMode.hex}, #f4fbff)` }}>{currentMode.glyph}</div>
            <div>
              <div className="ra-title">{currentMode.name}</div>
              <div className="ra-meta">
                <span className="ra-badge">{papers.length} papers indexed</span>
                <span className="ra-badge">{health.llm_available ? "LLM failover ready" : "Add API keys"}</span>
                {activeMode === "comparator" ? <span className="ra-badge" style={{ color: currentMode.hex }}>{selectedPaperIds.length}/3 selected</span> : null}
              </div>
            </div>
          </div>
          <div style={{ color: "#7083a9", fontSize: 12, textAlign: "right" }}>
            session id
            <div style={{ marginTop: 6, color: "#9cb0d4" }}>{clip(sid, 18)}</div>
          </div>
        </div>

        <div className="ra-sub">
          {activeMode === "reviewer" ? (
            <>
              <span>Select paper + lens, then run a live two-reviewer debate.</span>
              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <select className="ra-select" value={reviewPaperId} onChange={(event) => setReviewPaperId(event.target.value)}>
                  <option value="">Choose a paper</option>
                  {papers.map((paper) => <option key={paper.paper_id} value={paper.paper_id}>{paper.filename}</option>)}
                </select>
                <select className="ra-select" value={reviewLens} onChange={(event) => setReviewLens(event.target.value)} style={{ minWidth: 220 }}>
                  {REVIEW_LENSES.map((lens) => <option key={lens.id} value={lens.id}>{lens.label}</option>)}
                </select>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, color: reviewerDebug.turn_count >= 5 ? "#fbbf24" : "#9cb0d4", fontSize: 12 }}>
                <span>debate runway</span>
                <span style={{ letterSpacing: 2 }}>
                  {Array.from({ length: Number(reviewerDebug.max_turns || 8) }).map((_, idx) => (
                    <span key={idx}>{idx < Number(reviewerDebug.turn_count || 0) ? "*" : "o"}</span>
                  ))}
                </span>
                <span>
                  {Number(reviewerDebug.turn_count || 0) >= 7
                    ? "debate closing - final arguments"
                    : Number(reviewerDebug.turn_count || 0) >= 5
                      ? "debate entering caution window"
                      : ""}
                </span>
              </div>
            </>
          ) : activeMode === "comparator" ? (
            <>
              <span>Check up to three papers in the sidebar, then ask for a comparison.</span>
              <span style={{ color: currentMode.hex }}>{selectedPaperIds.length < 2 ? "Need at least 2 papers" : `${selectedPaperIds.length} papers selected`}</span>
            </>
          ) : (
            <>
              <span>{papers.length ? "Hybrid retrieval is live. Switch modes freely and keep the same conversation." : "Upload a paper to start semantic indexing."}</span>
              <span style={{ color: currentMode.hex }}>{health.indexed_papers || 0} indexed in backend</span>
            </>
          )}
        </div>

        <section className="ra-panel">
          <div className="ra-grid" />
          <div className="ra-scroll" ref={scrollRef}>
            <div className="ra-messages">
              {!history.length ? (
                <div className="ra-empty">
                  <div className="ra-empty-card">
                    <div className="ra-brand-badge" style={{ width: 66, height: 66, margin: "0 auto 18px", borderRadius: 22, fontSize: 20, background: `linear-gradient(135deg, ${currentMode.hex}, #f4fbff)` }}>{currentMode.glyph}</div>
                    <div className="ra-empty-title">{papers.length ? `Ask ${currentMode.name} something` : "Upload papers to begin"}</div>
                    <div className="ra-empty-text">
                      {papers.length
                        ? activeMode === "reviewer"
                          ? "Select a paper + lens, then run the debate. Optional prompts: 'skeptic: challenge novelty', 'advocate: defend method', 'vector 2', or 'next'."
                          : activeMode === "comparator"
                            ? "Choose two or three papers and compare their methods, benchmarks, results, or novelty."
                            : "Your papers are indexed with semantic chunks. Ask grounded questions, switch modes, and use one workspace for retrieval, writing, and review."
                        : "The backend can now serve the UI, store papers, and route chat through LangGraph. Upload one or more PDFs to start."}
                    </div>
                  </div>
                </div>
              ) : history.map((item) => {
                if (item.type === "divider") {
                  const mode = modeOf(item.mode);
                  return <div key={item.id} className="ra-divider"><div className="ra-line" /><div className="ra-divider-badge"><span>{mode.glyph}</span><span>switched to {mode.name}</span></div><div className="ra-line" /></div>;
                }
                const mode = modeOf(item.mode);
                const assistant = item.role === "assistant";
                return (
                  <div key={item.id} className={assistant ? "ra-assistant" : "ra-user"}>
                    {assistant ? <div className="ra-tag" style={{ color: mode.hex }}>{mode.glyph} {mode.name}</div> : null}
                    <div className={assistant ? "ra-assistant-bubble" : "ra-user-bubble"} style={!assistant ? { background: mode.hex } : null}>
                      {item.content}
                      {assistant && item.citations && item.citations.length ? (
                        <div className="ra-cites">
                          <div className="ra-cites-head">
                            <div className="ra-cites-title">References</div>
                            <div className="ra-cites-count">{item.citations.length}</div>
                          </div>
                          {item.citations.map((citation, index) => (
                            <div className="ra-cite" key={`${item.id}-${index}`}>
                              <div className="ra-cite-header">
                                <div className="ra-cite-label">{citation.filename || "Unknown source"}</div>
                                <div className="ra-cite-badge">[{index + 1}]</div>
                              </div>
                              <div className="ra-cite-block">
                                <div className="ra-cite-meta">
                                  {citation.page ? `Page ${citation.page}` : "Page n/a"}
                                  {citation.chunk_id ? ` | ${citation.chunk_id}` : ""}
                                </div>
                                <div className="ra-cite-text">{clip(citation.snippet, 260)}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })}
              {loading ? <div className="ra-dots">{[0,1,2].map((index) => <div key={index} className="ra-pulse" style={{ background: currentMode.hex, animationDelay: `${index * .15}s` }} />)}</div> : null}
              {error ? <div className="ra-error">{error}</div> : null}
            </div>
          </div>
        </section>

        <div className="ra-inputbar">
          <textarea
            className="ra-input"
            rows={1}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={activeMode === "reviewer" ? "Optional: ask skeptic/advocate, or type 'vector 2' / 'next'..." : activeMode === "comparator" ? "Ask how the selected papers differ in methods, benchmarks, or novelty..." : `Ask ${currentMode.name} something...`}
          />
          <button className="ra-send" type="button" disabled={loading || (activeMode === "reviewer" ? !reviewPaperId : !draft.trim())} style={{ background: currentMode.hex, opacity: (loading || (activeMode === "reviewer" ? !reviewPaperId : !draft.trim())) ? .5 : 1, cursor: (loading || (activeMode === "reviewer" ? !reviewPaperId : !draft.trim())) ? "not-allowed" : "pointer" }} onClick={send}>
            {loading ? "Thinking..." : activeMode === "reviewer" ? "Run Debate" : "Send"}
          </button>
        </div>
      </main>
    </div>
  );
}

if (typeof window !== "undefined" && window.ReactDOM) {
  ReactDOM.createRoot(document.getElementById("root")).render(<ResearchAgent />);
}

