// Trace lifecycle panel.
//
// Receives events forwarded by chat.js from the chat-completions SSE stream
// (where each ``hat_trace_event`` chunk represents one step in the wake
// pipeline) and renders them as a chronological timeline in the right-hand
// panel of the chat tab. Also seeds the panel with a session's existing
// traces on session open via ``GET /api/neocortex?session_id=...``.

import { $, escapeHtml } from "./util.js";
import { jget } from "./api.js";

const STAGE_BADGE = {
  uncertainty: "U",
  abstracting: "ABS",
  triage_start: "TRIAGE",
  triage_done: "TRIAGE",
  route_start: "ROUTE",
  route_done: "ROUTE",
  routed: "ROUTE",
  scored: "SCORE",
  created: "NEW",
  revised: "EDIT",
  rejected: "DROP",
  skipped: "SKIP",
  dropped: "DROP",
};

function panel() { return $("#trace-timeline"); }
function emptyMsg() { return $("#trace-empty"); }

function setEmpty(visible) {
  const el = emptyMsg();
  if (el) el.style.display = visible ? "" : "none";
}

function shortId(id) {
  if (!id) return "";
  return id.length > 10 ? id.slice(0, 10) + "…" : id;
}

function classifyStage(stage) {
  if (stage === "created") return "created";
  if (stage === "revised") return "revised";
  if (stage === "rejected" || stage === "dropped" || stage === "skipped") return "rejected";
  if (stage === "triage_start" || stage === "route_start" || stage === "abstracting") return "pending";
  return "info";
}

// Identifier for in-place card updates. ``*_start`` and the corresponding
// ``*_done`` share a key so the later event replaces the spinner card
// rather than appending a duplicate row.
function eventKey(ev) {
  const stage = ev.stage || "";
  const iid = ev.interaction_id || "";
  if (stage === "triage_start" || stage === "triage_done") return `triage:${iid}`;
  if (stage === "route_start" || stage === "route_done") return `route:${iid}`;
  if (stage === "uncertainty" || stage === "skipped") return `uncertainty:${iid}`;
  if (stage === "abstracting") return `abstracting:${iid}`;
  return null;
}

function renderCard(ev) {
  const stage = ev.stage || "event";
  const cls = classifyStage(stage);
  const li = document.createElement("li");
  li.className = `trace-event ${cls}`;
  if (ev.trace_id) li.dataset.traceId = ev.trace_id;
  const key = eventKey(ev);
  if (key) li.dataset.key = key;

  const badge = STAGE_BADGE[stage] || stage.slice(0, 4).toUpperCase();
  const tid = ev.trace_id ? shortId(ev.trace_id) : "";

  let body = "";
  if (stage === "uncertainty") {
    const u = typeof ev.uncertainty === "number" ? ev.uncertainty.toFixed(3) : "—";
    const thr = typeof ev.threshold === "number" ? ev.threshold.toFixed(2) : "—";
    body = `U=${u} (threshold ${thr})`;
  } else if (stage === "skipped") {
    const u = typeof ev.uncertainty === "number" ? ev.uncertainty.toFixed(3) : "—";
    const thr = typeof ev.threshold === "number" ? ev.threshold.toFixed(2) : "—";
    body = `gate skipped: U=${u} &lt; ${thr}`;
  } else if (stage === "triage_start") {
    body = `<em>running triage…</em>`;
  } else if (stage === "triage_done") {
    const keep = ev.keep;
    const verdict = keep === false ? "drop" : (keep === true ? "keep" : "?");
    const reason = ev.reason ? ` · ${escapeHtml(String(ev.reason))}` : "";
    body = `triage: <strong>${verdict}</strong>${reason}`;
  } else if (stage === "route_start") {
    const n = ev.n_priors || 0;
    body = `<em>routing… (${n} prior${n === 1 ? "" : "s"})</em>`;
  } else if (stage === "route_done") {
    const dec = ev.decision ? String(ev.decision).toUpperCase() : (ev.parsed ? "?" : "unparseable");
    const rationale = ev.rationale ? ` · ${escapeHtml(String(ev.rationale))}` : "";
    body = `route: <strong>${escapeHtml(dec)}</strong>${rationale}`;
  } else if (stage === "routed") {
    body = `Decision: <strong>${escapeHtml(ev.decision || "?")}</strong>`;
  } else if (stage === "scored") {
    const score = typeof ev.score === "number" ? ev.score.toFixed(2) : "—";
    const thr = typeof ev.threshold === "number" ? ev.threshold.toFixed(2) : "—";
    const accepted = ev.accepted ? "accept" : "reject";
    body = `score ${score} / ${thr} → ${accepted}`;
  } else if (stage === "created" || stage === "revised") {
    const tgt = (ev.target_response || "").trim();
    body = escapeHtml(tgt.length > 200 ? tgt.slice(0, 200) + "…" : tgt);
  } else if (stage === "rejected") {
    const score = typeof ev.score === "number" ? ev.score.toFixed(2) : "—";
    const thr = typeof ev.threshold === "number" ? ev.threshold.toFixed(2) : "—";
    body = `below threshold (score ${score} &lt; ${thr})`;
  } else if (stage === "dropped") {
    body = `abstractor dropped this turn`;
  } else if (stage === "abstracting") {
    const n = (ev.prior_trace_ids || []).length;
    body = n ? `Considering ${n} prior trace${n === 1 ? "" : "s"}…` : "Compressing turn…";
  }

  const rationale = ev.rationale && stage !== "route_done"
    ? `<div class="te-meta">${escapeHtml(ev.rationale)}</div>` : "";

  li.innerHTML = `
    <div class="te-head">
      <span class="te-badge">${escapeHtml(badge)}</span>
      <span class="te-id">${escapeHtml(tid)}</span>
    </div>
    <div class="te-body">${body}</div>
    ${rationale}
  `;
  return li;
}

export function appendTraceEvent(ev) {
  const list = panel();
  if (!list) return;
  setEmpty(false);
  const card = renderCard(ev);
  // If a card with the same key already exists (e.g. triage_start was
  // rendered earlier and now triage_done arrives), replace in place so
  // the timeline stays linear instead of growing duplicate "pending"
  // rows next to their resolved versions.
  const key = card.dataset.key;
  if (key) {
    const prev = list.querySelector(`li.trace-event[data-key="${CSS.escape(key)}"]`);
    if (prev) {
      list.replaceChild(card, prev);
      list.scrollTop = list.scrollHeight;
      return;
    }
  }
  list.appendChild(card);
  list.scrollTop = list.scrollHeight;
}

export function clearTracePanel() {
  const list = panel();
  if (!list) return;
  list.innerHTML = "";
  setEmpty(true);
}

export async function loadTracesForSession(sessionId) {
  clearTracePanel();
  if (!sessionId) return;
  try {
    const data = await jget(`/api/neocortex?session_id=${encodeURIComponent(sessionId)}`);
    const rows = data.data || [];
    if (!rows.length) return;
    for (const row of rows) {
      appendTraceEvent({
        stage: "created",
        trace_id: row.trace_id,
        target_response: row.response,
        rationale: (row.metadata && row.metadata.extras && row.metadata.extras.rationale) || "",
      });
    }
  } catch {
    // best-effort; panel stays empty
  }
}

export function initTracePanel() {
  const btn = $("#trace-clear-btn");
  if (btn) btn.addEventListener("click", clearTracePanel);
  setEmpty(true);
}
