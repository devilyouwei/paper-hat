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
  abstracting: "ABS",
  routed: "ROUTE",
  scored: "SCORE",
  created: "NEW",
  revised: "EDIT",
  rejected: "DROP",
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
  if (stage === "rejected") return "rejected";
  return "info";
}

function renderCard(ev) {
  const stage = ev.stage || "event";
  const cls = classifyStage(stage);
  const li = document.createElement("li");
  li.className = `trace-event ${cls}`;
  if (ev.trace_id) li.dataset.traceId = ev.trace_id;

  const badge = STAGE_BADGE[stage] || stage.slice(0, 4).toUpperCase();
  const tid = ev.trace_id ? shortId(ev.trace_id) : "";

  let body = "";
  if (stage === "routed") {
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
    body = `below threshold (score ${score} < ${thr})`;
  } else if (stage === "abstracting") {
    const n = (ev.prior_trace_ids || []).length;
    body = n ? `Considering ${n} prior trace${n === 1 ? "" : "s"}…` : "Compressing turn…";
  }

  const rationale = ev.rationale ? `<div class="te-meta">${escapeHtml(ev.rationale)}</div>` : "";

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
  // If this event refers to an existing trace_id and the latest visible
  // event for that trace is the same stage, collapse — otherwise append.
  const card = renderCard(ev);
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
