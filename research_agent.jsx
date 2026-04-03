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
  .ra-chip.selected { background: rgba(244,114,182,.14); border-color: rgba(244,114,182,.45); }
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
  .ra-sub-col { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
  .ra-inline-papers { display: flex; flex-wrap: wrap; gap: 8px; }
  .ra-inline-paper { display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(149,170,210,.22); background: rgba(20,30,51,.8); color: #c6d5f0; font-size: 12px; cursor: pointer; max-width: 320px; }
  .ra-inline-paper.selected { border-color: rgba(244,114,182,.6); background: rgba(244,114,182,.2); color: #ffe3f2; }
  .ra-inline-paper-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ra-inline-actions { display: flex; align-items: center; gap: 8px; }
  .ra-inline-btn { border: 1px solid rgba(149,170,210,.24); background: rgba(20,30,51,.88); color: #d7e4fb; border-radius: 999px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
  .ra-inline-btn:disabled { opacity: .45; cursor: not-allowed; }
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
  .ra-assistant-bubble.local { background: linear-gradient(165deg, rgba(14,24,40,.96), rgba(10,18,32,.98)); border: 1px solid rgba(96,165,250,.22); }
  .ra-assistant-bubble.reviewer { background: linear-gradient(160deg, rgba(27,22,14,.96), rgba(26,29,47,.96)); border: 1px solid rgba(251,146,60,.28); box-shadow: inset 0 0 0 1px rgba(251,146,60,.08); }
  .ra-assistant-bubble.reviewer .ra-md-h2 { color: #ffd9b3; }
  .ra-assistant-bubble.reviewer .ra-md-h3 { color: #ffe7ce; }
  .ra-assistant-bubble.reviewer .ra-md-p { color: #fde8d4; }
  .ra-assistant-bubble.reviewer .ra-md-ul li { color: #f7dcc0; }
  .ra-assistant-bubble.comparator { background: linear-gradient(165deg, rgba(12,20,34,.97), rgba(10,16,28,.99)); border: 1px solid rgba(120,146,199,.3); box-shadow: inset 0 0 0 1px rgba(120,146,199,.08); }
  .ra-assistant-bubble.comparator .ra-md-h2 { color: #e8f1ff; }
  .ra-assistant-bubble.comparator .ra-md-h3 { color: #f0f6ff; }
  .ra-assistant-bubble.comparator .ra-md-p { color: #d7e5ff; line-height: 1.78; }
  .ra-assistant-bubble.comparator .ra-md-ul li { color: #d9e5fb; }
  .ra-assistant.comparator-message { max-width: 96%; }
  .ra-comp-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
  .ra-comp-pill { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #d7e8ff; border: 1px solid rgba(120,146,199,.42); background: rgba(120,146,199,.16); border-radius: 999px; padding: 4px 10px; font-weight: 600; }
  .ra-comp-copy { border: 1px solid rgba(120,146,199,.45); background: rgba(120,146,199,.14); color: #eaf3ff; border-radius: 10px; font-size: 11px; padding: 5px 10px; cursor: pointer; font-weight: 600; }
  .ra-comp-copy:hover { background: rgba(120,146,199,.24); }
  .ra-comp-shell { display: flex; flex-direction: column; gap: 12px; }
  .ra-comp-preface { padding: 14px 14px 8px 14px; border-radius: 12px; border: 1px solid rgba(120,146,199,.24); background: rgba(17,27,44,.75); }
  .ra-comp-outline { display: flex; flex-wrap: wrap; gap: 8px; }
  .ra-comp-outline-chip { padding: 5px 10px; border-radius: 999px; border: 1px solid rgba(120,146,199,.3); background: rgba(120,146,199,.12); color: #dce9ff; font-size: 11px; }
  .ra-comp-sections { display: grid; grid-template-columns: 1fr; gap: 12px; }
  .ra-comp-section { border-radius: 14px; border: 1px solid rgba(120,146,199,.24); background: linear-gradient(160deg, rgba(19,30,47,.8), rgba(14,24,39,.82)); padding: 14px; box-shadow: 0 12px 24px rgba(0,0,0,.17); }
  .ra-comp-section.claim-matrix { border-left: 4px solid rgba(96,165,250,.72); }
  .ra-comp-section.conflict-map { border-left: 4px solid rgba(251,146,60,.72); }
  .ra-comp-section.benchmark-verdict-matrix { border-left: 4px solid rgba(52,211,153,.72); }
  .ra-comp-section.synthesis-blueprint { border-left: 4px solid rgba(244,114,182,.72); }
  .ra-comp-section.decision-by-use-case { border-left: 4px solid rgba(250,204,21,.72); }
  .ra-comp-section-head { display: flex; align-items: center; gap: 10px; margin-bottom: 11px; }
  .ra-comp-section-index { min-width: 32px; height: 24px; border-radius: 8px; display: grid; place-items: center; font-size: 11px; font-weight: 600; color: #d7e7ff; background: rgba(120,146,199,.16); border: 1px solid rgba(120,146,199,.36); }
  .ra-comp-section-title { margin: 0; font-size: 15px; color: #eef4ff; font-weight: 600; letter-spacing: .01em; }
  .ra-assistant-bubble.comparator .ra-md-h2 { margin: 0 0 8px 0; }
  .ra-assistant-bubble.comparator .ra-md-table-wrap { border-color: rgba(120,146,199,.28); background: rgba(17,28,44,.68); }
  .ra-assistant-bubble.comparator .ra-md-table th { background: rgba(31,47,71,.84); color: #edf4ff; }
  .ra-assistant-bubble.comparator .ra-md-table td { color: #dce9ff; }
  .ra-assistant-bubble.comparator .ra-md-table tbody tr:nth-child(even) td { background: rgba(24,37,57,.45); }
  .ra-review-live { margin-bottom: 12px; border-radius: 14px; border: 1px solid rgba(251,146,60,.3); background: rgba(39,26,14,.55); padding: 10px; }
  .ra-review-live-head { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; color: #ffd7ac; margin-bottom: 8px; }
  .ra-review-live-list { display: grid; gap: 8px; }
  .ra-review-turn { border-radius: 10px; border: 1px solid rgba(149,170,210,.22); background: rgba(11,17,31,.72); padding: 8px 10px; }
  .ra-review-turn.skeptic { border-color: rgba(251,146,60,.45); }
  .ra-review-turn.advocate { border-color: rgba(96,165,250,.45); }
  .ra-review-turn.judge { border-color: rgba(52,211,153,.45); }
  .ra-review-turn.synthesise { border-color: rgba(244,114,182,.45); }
  .ra-review-turn-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  .ra-review-turn-role { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; color: #f5dcc0; }
  .ra-review-turn-turn { font-size: 11px; color: #9cb0d4; }
  .ra-review-turn-text { font-size: 12px; line-height: 1.5; color: #e8f1ff; white-space: pre-wrap; }
  .ra-review-report { margin-bottom: 14px; border-radius: 16px; border: 1px solid rgba(251,146,60,.35); background: linear-gradient(145deg, rgba(55,33,13,.92), rgba(24,27,44,.94)); padding: 14px; box-shadow: 0 16px 28px rgba(0,0,0,.24); }
  .ra-review-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
  .ra-review-title { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; color: #ffd9b5; }
  .ra-review-chip { padding: 4px 9px; border-radius: 999px; border: 1px solid rgba(253,186,116,.4); background: rgba(253,186,116,.12); color: #ffd7aa; font-size: 11px; font-weight: 700; }
  .ra-review-overview { font-size: 13px; color: #fde7cf; line-height: 1.55; margin-bottom: 10px; }
  .ra-review-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; margin-bottom: 10px; }
  .ra-review-col { border-radius: 12px; border: 1px solid rgba(149,170,210,.22); background: rgba(15,20,35,.78); padding: 10px; }
  .ra-review-col h4 { margin: 0 0 8px 0; font-size: 12px; color: #e8f0ff; text-transform: uppercase; letter-spacing: .07em; }
  .ra-review-col ul { margin: 0; padding-left: 16px; }
  .ra-review-col li { margin: 0 0 6px 0; font-size: 12px; color: #cfddf8; line-height: 1.45; }
  .ra-review-decision { border-radius: 12px; border: 1px solid rgba(52,211,153,.35); background: rgba(11,41,34,.68); padding: 10px; }
  .ra-review-decision-title { margin: 0 0 6px 0; font-size: 12px; color: #b5f3dc; text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
  .ra-review-decision-text { font-size: 13px; color: #e6fff4; line-height: 1.5; }
  @media (max-width: 860px) { .ra-review-grid { grid-template-columns: 1fr; } }
  .ra-assistant-bubble .ra-md { white-space: normal; }
  .ra-md-h1, .ra-md-h2, .ra-md-h3 { margin: 0 0 10px 0; font-weight: 700; color: #f7fbff; line-height: 1.35; }
  .ra-md-h1 { font-size: 20px; }
  .ra-md-h2 { font-size: 18px; }
  .ra-md-h3 { font-size: 16px; }
  .ra-md-p { margin: 0 0 10px 0; color: #dce8ff; line-height: 1.6; }
  .ra-md-ul, .ra-md-ol { margin: 0 0 10px 20px; padding: 0; color: #dce8ff; }
  .ra-md-ul li, .ra-md-ol li { margin: 0 0 6px 0; line-height: 1.55; }
  .ra-md-table-wrap { margin: 0 0 12px 0; overflow-x: auto; border: 1px solid rgba(149,170,210,.2); border-radius: 12px; }
  .ra-md-table { width: 100%; min-width: 560px; border-collapse: collapse; background: rgba(10,16,30,.86); }
  .ra-md-table th, .ra-md-table td { border-bottom: 1px solid rgba(149,170,210,.16); padding: 9px 10px; text-align: left; vertical-align: top; font-size: 12px; line-height: 1.5; }
  .ra-md-table th { background: rgba(30,40,66,.9); color: #eef5ff; font-weight: 700; }
  .ra-md-table td { color: #d6e4ff; }
  .ra-md-table tr:last-child td { border-bottom: none; }
  .ra-md-strong { color: #ffffff; font-weight: 550; }
  .ra-assistant-bubble.comparator .ra-md-strong { color: #e6f0ff; font-weight: 500; }
  .ra-md-code { padding: 1px 6px; border-radius: 6px; background: rgba(96,165,250,.2); border: 1px solid rgba(96,165,250,.38); color: #cfe3ff; font-family: "IBM Plex Mono","Consolas",monospace; font-size: 12px; }
  .ra-md-math-inline { padding: 1px 6px; border-radius: 6px; border: 1px solid rgba(148,163,184,.36); background: rgba(15,23,42,.55); color: #edf4ff; display: inline-block; }
  .ra-md-math-block { margin: 0 0 12px 0; border-radius: 12px; border: 1px solid rgba(148,163,184,.32); background: linear-gradient(165deg, rgba(12,19,33,.92), rgba(9,15,27,.94)); padding: 12px; overflow-x: auto; }
  .ra-md-math-fallback { margin: 0; color: #e7f0ff; font-family: "Times New Roman","Cambria Math","IBM Plex Serif",serif; font-size: 16px; line-height: 1.5; white-space: pre-wrap; }
  .ra-md-math-inline-fallback { font-family: "Times New Roman","Cambria Math","IBM Plex Serif",serif; font-style: italic; font-size: 15px; }
  .ra-cites { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(145,164,203,.14); }
  .ra-cites-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .ra-cites-title { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #9cb3da; font-weight: 700; }
  .ra-cites-count { min-width: 22px; height: 22px; padding: 0 8px; border-radius: 999px; background: rgba(96,165,250,.14); border: 1px solid rgba(96,165,250,.34); color: #b9d4ff; font-size: 11px; display: grid; place-items: center; font-weight: 700; }
  .ra-cite { padding: 12px 14px; border-radius: 12px; background: rgba(10,16,30,.88); border: 1px solid rgba(145,164,203,.16); }
  .ra-cite-summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; cursor: pointer; list-style: none; }
  .ra-cite-summary::-webkit-details-marker { display: none; }
  .ra-cite-label { font-size: 12px; font-weight: 700; color: #e3eeff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ra-cite-badge { padding: 2px 7px; border-radius: 999px; background: rgba(96,165,250,.15); border: 1px solid rgba(96,165,250,.35); font-size: 10px; letter-spacing: .06em; color: #9ec6ff; font-weight: 700; }
  .ra-cite-meta { font-size: 11px; color: #95a9ce; white-space: nowrap; }
  .ra-cite-text { font-size: 12px; line-height: 1.55; color: #a9bddf; }
  .ra-cite-block { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
  .ra-comp-cites-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .ra-cite-group { border: 1px solid rgba(236,143,198,.24); border-radius: 12px; background: rgba(31,15,32,.52); padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  .ra-cite-group-head { font-size: 12px; font-weight: 700; color: #ffe4f4; border-bottom: 1px solid rgba(236,143,198,.2); padding-bottom: 6px; margin-bottom: 2px; }
  .ra-cite.comp { background: rgba(17,12,25,.72); border-color: rgba(236,143,198,.2); }
  @media (max-width: 860px) { .ra-comp-cites-grid { grid-template-columns: 1fr; } }
  .ra-error { padding: 12px 14px; border-radius: 14px; background: rgba(61,20,28,.9); border: 1px solid rgba(251,146,60,.35); color: #ffd0ad; font-size: 13px; }
  .ra-dots { display: flex; gap: 8px; padding-left: 24px; }
  .ra-pulse { width: 8px; height: 8px; border-radius: 999px; animation: raPulse 1.2s infinite ease-in-out; }
  .ra-inputbar { display: flex; align-items: flex-end; gap: 14px; padding: 16px 18px; border-radius: 24px; background: rgba(12,19,34,.92); border: 1px solid rgba(145,164,203,.12); }
  .ra-quickbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 2px -4px 2px; }
  .ra-quickbtn { border: 1px solid rgba(149,170,210,.24); background: rgba(20,30,51,.88); color: #d7e4fb; border-radius: 999px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
  .ra-quickbtn.active { border-color: rgba(244,114,182,.58); background: rgba(244,114,182,.2); color: #ffe3f2; }
  .ra-input { flex: 1; min-height: 56px; max-height: 120px; resize: none; padding: 16px 18px; border-radius: 18px; background: rgba(19,29,48,.95); color: #edf5ff; font: inherit; line-height: 1.5; }
  .ra-send { min-width: 120px; height: 56px; border-radius: 18px; color: #06101f; font-weight: 700; cursor: pointer; }
  @keyframes raPulse { 0%,80%,100% { transform: scale(.75); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
`;

const MODES = [
  { id: "local", name: "Local Brain", glyph: "L", hex: "#60a5fa", desc: "Answers only from your papers" },
  { id: "global", name: "Global Brain", glyph: "G", hex: "#a78bfa", desc: "Full reasoning plus paper context" },
  { id: "writer", name: "Paper Writer", glyph: "W", hex: "#34d399", desc: "Drafts in your research voice" },
  { id: "reviewer", name: "Reviewer", glyph: "R", hex: "#fb923c", desc: "Claim trial engine with judge + rewrite cards" },
  { id: "comparator", name: "Comparator", glyph: "C", hex: "#f472b6", desc: "Conflict map + verdict matrix across 2-3 papers" },
];
const REVIEW_LENSES = [
  { id: "full", label: "Full Review" },
  { id: "novelty", label: "Novelty" },
  { id: "method", label: "Methods" },
];
const COMPARATOR_QUICK_PROMPTS = [
  { id: "full", label: "Full Verdict", text: "Run a full comparator pass with claim matrix, conflict map, benchmark verdict matrix, and decision by use case." },
  { id: "conflict", label: "Conflict Map", text: "Focus on agreements, contradictions, and non-overlap across selected papers, then state what evidence resolves each conflict." },
  { id: "synthesis", label: "Synthesis Plan", text: "Build a synthesis blueprint that combines the strongest parts of each paper and proposes one merged experiment." },
];
function comparatorPromptById(id) { return (COMPARATOR_QUICK_PROMPTS.find((preset) => preset.id === id) || COMPARATOR_QUICK_PROMPTS[0]).text; }
function comparatorLabelById(id) { return (COMPARATOR_QUICK_PROMPTS.find((preset) => preset.id === id) || COMPARATOR_QUICK_PROMPTS[0]).label; }

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
function hasReviewerConversation(history) {
  return (history || []).some((item) => item.role === "assistant" && item.mode === "reviewer");
}
function reviewerReportCompleted(debug) {
  return !!(debug && debug.final_report_ready);
}
function hasReviewerCompletedConversation(history) {
  return (history || []).some(
    (item) =>
      item.role === "assistant" &&
      item.mode === "reviewer" &&
      (
        (item.debug && item.debug.final_report_ready) ||
        String(item.content || "").includes("## Reviewer Complete Report")
      ),
  );
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

function renderLatex(expression, { displayMode = false, key } = {}) {
  const latex = String(expression || "").trim();
  if (!latex) return null;
  const katex = typeof window !== "undefined" ? window.katex : null;
  if (katex && typeof katex.renderToString === "function") {
    try {
      const html = katex.renderToString(latex, { displayMode, throwOnError: false, strict: "ignore" });
      if (displayMode) {
        return <div key={key} className="ra-md-math-block" dangerouslySetInnerHTML={{ __html: html }} />;
      }
      return <span key={key} className="ra-md-math-inline" dangerouslySetInnerHTML={{ __html: html }} />;
    } catch (_error) {
      // Fall through to graceful text fallback.
    }
  }

  if (displayMode) {
    return (
      <div key={key} className="ra-md-math-block">
        <pre className="ra-md-math-fallback">{latex}</pre>
      </div>
    );
  }
  return <code key={key} className="ra-md-math-inline ra-md-math-inline-fallback">{latex}</code>;
}

function renderInlineSegments(text, keyPrefix = "seg") {
  const raw = String(text || "");
  const tokens = raw.split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$\n]+\$)/g).filter(Boolean);
  return tokens.map((token, index) => {
    if (/^\*\*[^*]+\*\*$/.test(token)) {
      return <strong key={`${keyPrefix}-b-${index}`} className="ra-md-strong">{token.slice(2, -2)}</strong>;
    }
    if (/^`[^`]+`$/.test(token)) {
      return <code key={`${keyPrefix}-c-${index}`} className="ra-md-code">{token.slice(1, -1)}</code>;
    }
    if (/^\$[^$\n]+\$$/.test(token)) {
      return renderLatex(token.slice(1, -1), { displayMode: false, key: `${keyPrefix}-m-${index}` });
    }
    return <React.Fragment key={`${keyPrefix}-t-${index}`}>{token}</React.Fragment>;
  });
}

function renderAssistantMarkdown(content) {
  const lines = String(content || "").split("\n");
  const blocks = [];
  let listType = null;
  let listItems = [];
  let tableRows = [];
  let inMathBlock = false;
  let mathLines = [];
  let mathDelimiter = "$$";
  let key = 0;

  const flushList = () => {
    if (!listType || !listItems.length) return;
    const Tag = listType === "ol" ? "ol" : "ul";
    blocks.push(
      <Tag key={`list-${key++}`} className={listType === "ol" ? "ra-md-ol" : "ra-md-ul"}>
        {listItems.map((item, idx) => <li key={`li-${idx}`}>{renderInlineSegments(item, `li-${idx}`)}</li>)}
      </Tag>,
    );
    listType = null;
    listItems = [];
  };

  const parseTableCells = (row) =>
    String(row || "")
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());

  const isSeparatorRow = (cells) =>
    cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));

  const flushTable = () => {
    if (!tableRows.length) return;
    const parsedRows = tableRows.map(parseTableCells).filter((cells) => cells.length > 0);
    tableRows = [];
    if (!parsedRows.length) return;
    const rows = parsedRows.filter((cells) => !isSeparatorRow(cells));
    if (!rows.length) return;

    const header = rows[0];
    const bodyRows = rows.slice(1);
    blocks.push(
      <div key={`tbl-wrap-${key++}`} className="ra-md-table-wrap">
        <table className="ra-md-table">
          <thead>
            <tr>
              {header.map((cell, index) => (
                <th key={`th-${index}`}>{renderInlineSegments(cell, `th-${index}`)}</th>
              ))}
            </tr>
          </thead>
          {bodyRows.length ? (
            <tbody>
              {bodyRows.map((row, rowIndex) => (
                <tr key={`tr-${rowIndex}`}>
                  {header.map((_, columnIndex) => (
                    <td key={`td-${rowIndex}-${columnIndex}`}>
                      {renderInlineSegments(row[columnIndex] || "", `td-${rowIndex}-${columnIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          ) : null}
        </table>
      </div>,
    );
  };

  const flushMathBlock = () => {
    if (!mathLines.length) return;
    const latex = mathLines.join("\n").trim();
    mathLines = [];
    if (!latex) return;
    blocks.push(renderLatex(latex, { displayMode: true, key: `math-${key++}` }));
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (inMathBlock) {
      if (
        (mathDelimiter === "$$" && trimmed === "$$")
        || (mathDelimiter === "\\[" && trimmed === "\\]")
      ) {
        flushMathBlock();
        inMathBlock = false;
        mathDelimiter = "$$";
      } else {
        mathLines.push(line);
      }
      continue;
    }

    const singleLineDollarMath = trimmed.match(/^\$\$(.+)\$\$$/);
    const singleLineBracketMath = trimmed.match(/^\\\[(.+)\\\]$/);
    if (singleLineDollarMath || singleLineBracketMath) {
      flushList();
      flushTable();
      const latex = (singleLineDollarMath ? singleLineDollarMath[1] : singleLineBracketMath[1]).trim();
      if (latex) {
        blocks.push(renderLatex(latex, { displayMode: true, key: `math-single-${key++}` }));
      }
      continue;
    }

    if (trimmed === "$$" || trimmed === "\\[") {
      flushList();
      flushTable();
      inMathBlock = true;
      mathDelimiter = trimmed === "\\[" ? "\\[" : "$$";
      mathLines = [];
      continue;
    }

    if (!trimmed) {
      flushList();
      flushTable();
      continue;
    }

    if (/^\|.*\|$/.test(trimmed)) {
      flushList();
      tableRows.push(trimmed);
      continue;
    }

    flushTable();

    const orderedMatch = trimmed.match(/^\s*(\d+)\.\s+(.+)/);
    if (orderedMatch) {
      if (listType && listType !== "ol") flushList();
      listType = "ol";
      listItems.push(orderedMatch[2]);
      continue;
    }

    const unorderedMatch = trimmed.match(/^\s*[-*+]\s+(.+)/);
    if (unorderedMatch) {
      if (listType && listType !== "ul") flushList();
      listType = "ul";
      listItems.push(unorderedMatch[1]);
      continue;
    }

    flushList();

    if (trimmed.startsWith("### ")) {
      blocks.push(<h3 key={`h3-${key++}`} className="ra-md-h3">{renderInlineSegments(trimmed.slice(4), `h3-${key}`)}</h3>);
      continue;
    }
    if (trimmed.startsWith("## ")) {
      blocks.push(<h2 key={`h2-${key++}`} className="ra-md-h2">{renderInlineSegments(trimmed.slice(3), `h2-${key}`)}</h2>);
      continue;
    }
    if (trimmed.startsWith("# ")) {
      blocks.push(<h1 key={`h1-${key++}`} className="ra-md-h1">{renderInlineSegments(trimmed.slice(2), `h1-${key}`)}</h1>);
      continue;
    }
    blocks.push(<p key={`p-${key++}`} className="ra-md-p">{renderInlineSegments(trimmed, `p-${key}`)}</p>);
  }

  flushMathBlock();
  flushList();
  flushTable();
  return <div className="ra-md">{blocks.length ? blocks : <p className="ra-md-p">{content}</p>}</div>;
}

const COMPARATOR_SECTION_TITLES = [
  "Papers Compared",
  "Claim Matrix",
  "Conflict Map",
  "Benchmark Verdict Matrix",
  "Method Trade-offs",
  "Synthesis Blueprint",
  "Decision By Use Case",
];

function normalizeComparatorMarkdown(content) {
  let normalized = String(content || "").replace(/\r\n/g, "\n");
  for (const title of COMPARATOR_SECTION_TITLES) {
    const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const linePattern = new RegExp(`^\\s*${escaped}\\b\\s*:?(.*)$`, "gmi");
    normalized = normalized.replace(linePattern, (full, tail) => {
      const trimmed = String(full || "").trim();
      if (trimmed.startsWith("#")) return full;
      const rest = String(tail || "").trim();
      return rest ? `## ${title}\n${rest}` : `## ${title}`;
    });
  }
  return normalized;
}

function splitComparatorSections(content) {
  const lines = String(content || "").split("\n");
  const sections = [];
  const introLines = [];
  let currentTitle = "";
  let currentLines = [];

  const flushSection = () => {
    if (!currentTitle) return;
    sections.push({
      title: currentTitle,
      body: currentLines.join("\n").trim(),
    });
    currentTitle = "";
    currentLines = [];
  };

  for (const line of lines) {
    const headingMatch = line.trim().match(/^##\s+(.+)$/);
    if (headingMatch) {
      flushSection();
      currentTitle = headingMatch[1].trim();
      continue;
    }
    if (currentTitle) {
      currentLines.push(line);
    } else {
      introLines.push(line);
    }
  }
  flushSection();
  return {
    intro: introLines.join("\n").trim(),
    sections,
  };
}

function comparatorSectionClass(title) {
  const slug = String(title || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || "general";
}

function renderComparatorAnswer(content) {
  const normalized = normalizeComparatorMarkdown(content);
  const { intro, sections } = splitComparatorSections(normalized);
  if (!sections.length) {
    return renderAssistantMarkdown(normalized);
  }

  return (
    <div className="ra-comp-shell">
      {intro ? <div className="ra-comp-preface">{renderAssistantMarkdown(intro)}</div> : null}
      <div className="ra-comp-outline">
        {sections.map((section, index) => (
          <span key={`outline-${index}`} className="ra-comp-outline-chip">
            {index + 1}. {section.title}
          </span>
        ))}
      </div>
      <div className="ra-comp-sections">
        {sections.map((section, index) => (
          <section key={`sec-${index}`} className={`ra-comp-section ${comparatorSectionClass(section.title)}`}>
            <div className="ra-comp-section-head">
              <span className="ra-comp-section-index">{String(index + 1).padStart(2, "0")}</span>
              <h3 className="ra-comp-section-title">{section.title}</h3>
            </div>
            {renderAssistantMarkdown(section.body || "No detail provided.")}
          </section>
        ))}
      </div>
    </div>
  );
}

function groupCitationsByFilename(citations) {
  const grouped = new Map();
  (citations || []).forEach((citation, index) => {
    const filename = String(citation?.filename || "Unknown source");
    if (!grouped.has(filename)) {
      grouped.set(filename, []);
    }
    grouped.get(filename).push({ citation, index });
  });
  return Array.from(grouped.entries()).map(([filename, entries]) => ({ filename, entries }));
}

function renderReviewerFinalReportCard(report) {
  if (!report || typeof report !== "object") return null;
  const commonPoints = Array.isArray(report.common_points) ? report.common_points.filter(Boolean) : [];
  const confidence = Number.isFinite(Number(report.confidence)) ? Number(report.confidence) : null;

  return (
    <div className="ra-review-report">
      <div className="ra-review-head">
        <div className="ra-review-title">Panel Snapshot</div>
        <div className="ra-review-chip">{confidence !== null ? `confidence ${(confidence * 100).toFixed(0)}%` : "executive summary"}</div>
      </div>
      <div className="ra-review-overview">{report.overview || "Final panel summary is ready."}</div>
      <div className="ra-review-col" style={{ marginBottom: 10 }}>
        <h4>Common Points</h4>
        <ul>
          {(commonPoints.length ? commonPoints : ["No common points captured."]).map((item, idx) => <li key={`cp-${idx}`}>{item}</li>)}
        </ul>
      </div>
      <div className="ra-review-grid" style={{ marginBottom: 10 }}>
        <div className="ra-review-col">
          <h4>Skeptic Conclusion</h4>
          <div className="ra-review-overview">{report.skeptic_conclusion || "Not available."}</div>
        </div>
        <div className="ra-review-col">
          <h4>Advocate Conclusion</h4>
          <div className="ra-review-overview">{report.advocate_conclusion || "Not available."}</div>
        </div>
      </div>
      <div className="ra-review-col" style={{ marginBottom: 10 }}>
        <h4>Joint Conclusion</h4>
        <div className="ra-review-overview">{report.joint_conclusion || "Not available."}</div>
      </div>
      <div className="ra-review-decision">
        <div className="ra-review-decision-title">Final Decision</div>
        <div className="ra-review-decision-text">{report.final_decision || "Decision not available."}</div>
      </div>
    </div>
  );
}

function normalizeRoundEvents(debug) {
  const raw = Array.isArray(debug?.round_events) ? debug.round_events : [];
  return raw
    .map((event) => ({
      speaker: String(event?.speaker || "").trim().toLowerCase(),
      vector_id: String(event?.vector_id || "").trim(),
      turn: Number(event?.turn || 0),
      content: String(event?.content || "").trim(),
    }))
    .filter((event) => ["skeptic", "advocate", "judge", "synthesise"].includes(event.speaker) && event.content);
}

function reviewerSpeakerLabel(speaker) {
  if (speaker === "skeptic") return "Skeptic";
  if (speaker === "advocate") return "Advocate";
  if (speaker === "judge") return "Judge";
  if (speaker === "synthesise") return "Rewrite Compiler";
  return "Panel";
}

function renderReviewerRoundEvents(debug) {
  const events = normalizeRoundEvents(debug);
  if (!events.length) return null;
  return (
    <div className="ra-review-live">
      <div className="ra-review-live-head">Live Debate Round</div>
      <div className="ra-review-live-list">
        {events.map((event, index) => (
          <div key={`round-${event.speaker}-${event.turn || index}-${index}`} className={`ra-review-turn ${event.speaker}`}>
            <div className="ra-review-turn-head">
              <div className="ra-review-turn-role">{reviewerSpeakerLabel(event.speaker)}</div>
              <div className="ra-review-turn-turn">{event.turn ? `Turn ${event.turn}` : "Structured event"}</div>
            </div>
            <div className="ra-review-turn-text">{renderInlineSegments(event.content, `round-${index}`)}</div>
          </div>
        ))}
      </div>
    </div>
  );
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
  const [comparatorPreset, setComparatorPreset] = useState("full");
  const reviewInterventionMode = "ask";
  const [history, setHistory] = useState([]);
  const [styleProfile, setStyleProfile] = useState({ active: false, profile: "", source_count: 0 });
  const [health, setHealth] = useState({ llm_available: false, indexed_papers: 0 });
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [copiedComparatorId, setCopiedComparatorId] = useState("");
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
  const reviewerCompleted = useMemo(
    () => reviewerReportCompleted(reviewerDebug) || hasReviewerCompletedConversation(history),
    [reviewerDebug, history],
  );

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
    setHistory((current) => current.concat({ type: "divider", id: `${Date.now()}-${modeId}`, mode: modeId }));
  }

  function toggleComparator(paperId) {
    setSelectedPaperIds((current) => {
      if (current.includes(paperId)) return current.filter((id) => id !== paperId);
      if (current.length >= 3) return current;
      return current.concat(paperId);
    });
  }

  function selectTopTwoForComparator() {
    if (papers.length < 2) return;
    setSelectedPaperIds(papers.slice(0, 2).map((paper) => paper.paper_id));
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

  async function send(reviewerAction = "auto") {
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
    const comparatorMessage = comparatorPromptById(comparatorPreset);
    const reviewerHasHistory = hasReviewerConversation(history);
    const reviewerHasComplete = hasReviewerCompletedConversation(history);
    let message = "";
    let userVisibleMessage = "";
    if (activeMode === "reviewer") {
      if (reviewerAction === "restart" || !reviewerHasHistory || reviewerHasComplete) {
        const lens = reviewLensLabel(reviewLens);
        message = `[Start Debate] Focus lens: ${lens}`;
        userVisibleMessage = `[Start Debate] ${lens}`;
      } else {
        message = "next";
        userVisibleMessage = "next";
      }
    } else if (activeMode === "comparator") {
      message = comparatorMessage;
      userVisibleMessage = `[Compare: ${comparatorLabelById(comparatorPreset)}]`;
    } else {
      message = rawMessage;
      userVisibleMessage = rawMessage;
    }
    if (!message) return;

    const historyPayload = activeMode === "comparator" ? [] : compactHistory();
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

  async function copyComparatorAnswer(messageId, content) {
    const text = String(content || "").trim();
    if (!text) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const probe = document.createElement("textarea");
        probe.value = text;
        probe.setAttribute("readonly", "true");
        probe.style.position = "absolute";
        probe.style.left = "-9999px";
        document.body.appendChild(probe);
        probe.select();
        document.execCommand("copy");
        document.body.removeChild(probe);
      }
      setCopiedComparatorId(messageId);
      setTimeout(() => {
        setCopiedComparatorId((current) => (current === messageId ? "" : current));
      }, 1400);
    } catch (err) {}
  }

  return (
    <div className="ra-app">
      <aside className="ra-side">
        <div className="ra-brand">
          <div className="ra-brand-row"><div className="ra-brand-badge">RA</div><div>Research Agent</div></div>
          <div className="ra-subtitle">Pinecone + Gemini + Groq + OpenRouter</div>
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
                className={`ra-chip${selected ? " selected" : ""}`}
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
              <span>Select paper + lens, then run a claim trial (skeptic vs advocate + evidence-only judge).</span>
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
              <div className="ra-sub-col" style={{ flex: 1 }}>
                <span>Choose exactly which papers to compare (2-3).</span>
                <div className="ra-inline-papers">
                  {papers.map((paper) => {
                    const selected = selectedPaperIds.includes(paper.paper_id);
                    const disabled = !selected && selectedPaperIds.length >= 3;
                    return (
                      <button
                        key={`cmp-top-${paper.paper_id}`}
                        type="button"
                        className={`ra-inline-paper${selected ? " selected" : ""}`}
                        disabled={disabled}
                        onClick={() => toggleComparator(paper.paper_id)}
                        title={paper.filename}
                      >
                        <input
                          type="checkbox"
                          checked={selected}
                          readOnly
                          onClick={(event) => event.stopPropagation()}
                        />
                        <span className="ra-inline-paper-name">{paper.filename}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
              <div className="ra-inline-actions">
                <button
                  type="button"
                  className="ra-inline-btn"
                  onClick={() => setSelectedPaperIds([])}
                  disabled={!selectedPaperIds.length}
                >
                  Clear
                </button>
                <button
                  type="button"
                  className="ra-inline-btn"
                  onClick={selectTopTwoForComparator}
                  disabled={papers.length < 2}
                >
                  Pick first 2
                </button>
                <span style={{ color: currentMode.hex, fontWeight: 700, minWidth: 120, textAlign: "right" }}>
                  {selectedPaperIds.length < 2 ? "Need 2+" : `${selectedPaperIds.length} selected`}
                </span>
              </div>
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
                          ? "Select a paper + lens, then use Start Debate and Next Turn to run the structured reviewer panel."
                          : activeMode === "comparator"
                            ? "Choose two or three papers, then run conflict mapping and a verdict matrix."
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
                const bubbleClass = assistant
                  ? `ra-assistant-bubble${item.mode === "local" ? " local" : ""}${item.mode === "reviewer" ? " reviewer" : ""}${item.mode === "comparator" ? " comparator" : ""}`
                  : "ra-user-bubble";
                const comparatorCiteGroups = assistant && item.mode === "comparator"
                  ? groupCitationsByFilename(item.citations || [])
                  : [];
                return (
                  <div key={item.id} className={assistant ? `ra-assistant${item.mode === "comparator" ? " comparator-message" : ""}` : "ra-user"}>
                    {assistant ? <div className="ra-tag" style={{ color: mode.hex }}>{mode.glyph} {mode.name}</div> : null}
                    <div className={bubbleClass} style={!assistant ? { background: mode.hex } : null}>
                      {assistant && item.mode === "comparator" ? (
                        <div className="ra-comp-toolbar">
                          <div className="ra-comp-pill">Comparator Verdict</div>
                          <button className="ra-comp-copy" type="button" onClick={() => copyComparatorAnswer(item.id, item.content)}>
                            {copiedComparatorId === item.id ? "Copied" : "Copy Answer"}
                          </button>
                        </div>
                      ) : null}
                      {assistant && item.mode === "reviewer" && item.debug ? renderReviewerRoundEvents(item.debug) : null}
                      {assistant && item.mode === "reviewer" && item.debug && item.debug.final_report ? renderReviewerFinalReportCard(item.debug.final_report) : null}
                      {assistant
                        ? (item.mode === "comparator" ? renderComparatorAnswer(item.content) : renderAssistantMarkdown(item.content))
                        : item.content}
                      {assistant && item.citations && item.citations.length ? (
                        <div className="ra-cites">
                          <div className="ra-cites-head">
                            <div className="ra-cites-title">{item.mode === "comparator" ? "Evidence" : "References"}</div>
                            <div className="ra-cites-count">{item.citations.length}</div>
                          </div>
                          {item.mode === "comparator" ? (
                            <div className="ra-comp-cites-grid">
                              {comparatorCiteGroups.map((group, groupIndex) => (
                                <div className="ra-cite-group" key={`${item.id}-group-${groupIndex}`}>
                                  <div className="ra-cite-group-head">{group.filename}</div>
                                  {group.entries.map(({ citation, index }) => (
                                    <details className="ra-cite comp" key={`${item.id}-${groupIndex}-${index}`}>
                                      <summary className="ra-cite-summary">
                                        <div className="ra-cite-label">[{index + 1}] {citation.page ? `p.${citation.page}` : "page n/a"}</div>
                                        <div className="ra-cite-meta">{citation.chunk_id ? clip(citation.chunk_id, 28) : "snippet"}</div>
                                      </summary>
                                      <div className="ra-cite-block">
                                        <div className="ra-cite-text">{clip(citation.snippet, 360)}</div>
                                      </div>
                                    </details>
                                  ))}
                                </div>
                              ))}
                            </div>
                          ) : item.citations.map((citation, index) => (
                            <details className="ra-cite" key={`${item.id}-${index}`}>
                              <summary className="ra-cite-summary">
                                <div className="ra-cite-label">[{index + 1}] {citation.filename || "Unknown source"}</div>
                                <div className="ra-cite-meta">{citation.page ? `p.${citation.page}` : "page n/a"}</div>
                              </summary>
                              <div className="ra-cite-block">
                                {citation.chunk_id ? <div className="ra-cite-meta">chunk {clip(citation.chunk_id, 40)}</div> : null}
                                <div className="ra-cite-text">{clip(citation.snippet, 360)}</div>
                              </div>
                            </details>
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

        {activeMode === "comparator" ? (
          <div className="ra-quickbar">
            {COMPARATOR_QUICK_PROMPTS.map((preset) => (
              <button
                key={`cmp-quick-${preset.id}`}
                type="button"
                className={`ra-quickbtn${comparatorPreset === preset.id ? " active" : ""}`}
                onClick={() => setComparatorPreset(preset.id)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        ) : null}

        {activeMode === "comparator" ? (
          <div className="ra-inputbar">
            <div style={{ flex: 1, minHeight: 56, display: "flex", flexDirection: "column", justifyContent: "center", gap: 4, padding: "0 4px" }}>
              <div style={{ fontSize: 12, color: "#8ea3c8", textTransform: "uppercase", letterSpacing: ".08em" }}>Comparison Focus</div>
              <div style={{ fontSize: 14, color: "#eaf2ff", fontWeight: 700 }}>{comparatorLabelById(comparatorPreset)}</div>
            </div>
            <button className="ra-send" type="button" disabled={loading || selectedPaperIds.length < 2} style={{ background: currentMode.hex, opacity: (loading || selectedPaperIds.length < 2) ? .5 : 1, cursor: (loading || selectedPaperIds.length < 2) ? "not-allowed" : "pointer" }} onClick={() => send()}>
              {loading ? "Thinking..." : "Compare"}
            </button>
          </div>
        ) : activeMode === "reviewer" ? (
          <div className="ra-inputbar">
            <div style={{ flex: 1, minHeight: 56, display: "flex", flexDirection: "column", justifyContent: "center", gap: 4, padding: "0 4px" }}>
              <div style={{ fontSize: 12, color: "#8ea3c8", textTransform: "uppercase", letterSpacing: ".08em" }}>Reviewer Controls</div>
              <div style={{ fontSize: 14, color: "#eaf2ff", fontWeight: 700 }}>
                {reviewerCompleted
                  ? "Report is complete. Start a new debate if you want another run."
                  : hasReviewerConversation(history)
                    ? "Continue structured debate (no free-text input)."
                    : "Start structured debate from selected lens."}
              </div>
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              {reviewerCompleted ? (
                <button
                  className="ra-send"
                  type="button"
                  disabled={loading || !reviewPaperId}
                  style={{ background: currentMode.hex, opacity: (loading || !reviewPaperId) ? .5 : 1, cursor: (loading || !reviewPaperId) ? "not-allowed" : "pointer" }}
                  onClick={() => send("restart")}
                >
                  {loading ? "Thinking..." : "Start New Debate"}
                </button>
              ) : (
                <>
                  <button
                    className="ra-send"
                    type="button"
                    disabled={loading || !reviewPaperId}
                    style={{ background: currentMode.hex, opacity: (loading || !reviewPaperId) ? .5 : 1, cursor: (loading || !reviewPaperId) ? "not-allowed" : "pointer" }}
                    onClick={() => send(hasReviewerConversation(history) ? "next" : "start")}
                  >
                    {loading ? "Thinking..." : (hasReviewerConversation(history) ? "Next Turn" : "Start Debate")}
                  </button>
                  <button
                    className="ra-send"
                    type="button"
                    disabled={loading || !reviewPaperId}
                    style={{ background: "#8ea3c8", color: "#0b1322", minWidth: 132, opacity: (loading || !reviewPaperId) ? .5 : 1, cursor: (loading || !reviewPaperId) ? "not-allowed" : "pointer" }}
                    onClick={() => send("restart")}
                  >
                    Restart
                  </button>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="ra-inputbar">
            <textarea
              className="ra-input"
              rows={1}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder={activeMode === "reviewer" ? "Optional: ask skeptic/advocate, or type 'vector 2' / 'next'..." : `Ask ${currentMode.name} something...`}
            />
            <button className="ra-send" type="button" disabled={loading || !draft.trim()} style={{ background: currentMode.hex, opacity: (loading || !draft.trim()) ? .5 : 1, cursor: (loading || !draft.trim()) ? "not-allowed" : "pointer" }} onClick={() => send()}>
              {loading ? "Thinking..." : "Send"}
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

if (typeof window !== "undefined" && window.ReactDOM) {
  ReactDOM.createRoot(document.getElementById("root")).render(<ResearchAgent />);
}

