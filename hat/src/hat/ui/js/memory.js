import { $, escapeHtml } from "./util.js";
import { jget, jpatch, jdelete } from "./api.js";

let memEntries = [];
let memEditingId = null;
let memQuery = "";
// Live write-policy config (threshold). Loaded from /api/policy on first
// refresh and reused for the breakdown view.
let memPolicy = null;

function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function memRowHtml(e, i) {
  const score = fmt(e.score);
  const sig = e.signals || {};
  const u = fmt(sig.uncertainty);
  const md = e.metadata || {};
  const extras = (md && md.extras) || {};
  const reason = escapeHtml(extras.route_reason || "");
  const q = escapeHtml((e.query || "").slice(0, 100));
  const r = escapeHtml((e.response || "").slice(0, 200));
  const isOracle =
    (md.extras && md.extras.oracle) ||
    (typeof md.source === "string" && md.source.includes("oracle"));
  const oracleBadge = isOracle
    ? `<span class="badge oracle" title="${escapeHtml(
        (md.extras && md.extras.oracle_name) || "oracle",
      )}">oracle</span>`
    : "";
  return `<tr data-id="${e.trace_id}">
    <td>${i + 1}</td>
    <td><code>${escapeHtml((e.trace_id || "").slice(0, 8))}</code>${oracleBadge}</td>
    <td><strong>${score}</strong></td>
    <td>${u}</td>
    <td class="truncate" title="${reason}">${reason}</td>
    <td class="truncate" title="${escapeHtml(e.query || "")}">${q}</td>
    <td class="truncate" title="${escapeHtml(e.response || "")}">${r}</td>
    <td class="actions">
      <button class="small" data-act="edit">Edit</button>
      <button class="small danger" data-act="del">Delete</button>
    </td>
  </tr>`;
}

function renderMemRows() {
  const q = memQuery.trim().toLowerCase();
  const filtered = q
    ? memEntries.filter(
        (e) =>
          (e.query || "").toLowerCase().includes(q) ||
          (e.response || "").toLowerCase().includes(q),
      )
    : memEntries;
  const tbody = $("#mem-table tbody");
  if (filtered.length) {
    tbody.innerHTML = filtered.map(memRowHtml).join("");
  } else {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">No entries match.</td></tr>`;
  }
  $("#mem-count").textContent = `${filtered.length} / ${memEntries.length}`;
}

function renderPolicy() {
  if (!memPolicy) return;
  const { threshold } = memPolicy;
  const t = fmt(threshold);
  const summary = $("#mem-policy-summary");
  if (summary) summary.textContent = `score = U · threshold ${t}`;
  const formula = $("#mem-formula");
  if (formula) formula.textContent =
    `score(m) = U(m)    ·    write if U ≥ ${t}`;
  if (memPolicy.oracle) {
    const o = memPolicy.oracle;
    const el = $("#mem-oracle-summary");
    if (!el) return;
    if (o.enabled) {
      el.textContent =
        `Oracle: ${o.model} consulted when U > ${fmt(o.threshold)}. Limits: ${fmt(o.rps, 2)}/s, ${o.daily_calls}/day.`;
    } else {
      el.textContent =
        "Oracle: disabled (set HAT_ORACLE_ENABLED=true to consult an external teacher when the cortex is unsure).";
    }
  }
}

async function loadPolicy() {
  // The policy rarely changes (env-driven, restart to override) so we cache
  // it for the lifetime of the tab. Failure is non-fatal: the row view
  // falls back to the paper defaults α=β=0.4, γ=0.2.
  if (memPolicy) return;
  try {
    const data = await jget("/api/policy");
    memPolicy = { ...(data.write_policy || {}), oracle: data.oracle };
    renderPolicy();
  } catch (e) {
    // ignore; defaults are fine for visualisation
  }
}

export async function refreshMemory() {
  $("#mem-status").textContent = "loading…";
  await loadPolicy();
  try {
    const data = await jget("/api/neocortex");
    memEntries = data.data || [];
    renderMemRows();
    $("#mem-status").textContent = `${memEntries.length} entries`;
  } catch (e) {
    $("#mem-status").textContent = `load failed: ${e.message}`;
  }
}

// Cache of session details (id -> {session, messages: Interaction[]}). The
// memory editor uses this to render where each trace came from without
// re-fetching on every open. Cache is keyed by session_id; entries are
// loaded lazily and dropped when the tab is left.
const sessionCache = new Map();

async function fetchSession(sessionId) {
  if (sessionCache.has(sessionId)) return sessionCache.get(sessionId);
  try {
    const detail = await jget(`/api/sessions/${sessionId}`);
    sessionCache.set(sessionId, detail);
    return detail;
  } catch (e) {
    return null;
  }
}

async function renderSource(e) {
  const body = $("#mem-source-body");
  const summary = $("#mem-source-summary");
  if (!body) return;
  const sid = e.session_id;
  const iids = e.interaction_ids || [];
  if (summary) {
    summary.textContent = sid
      ? `session ${sid.slice(0, 8)} · ${iids.length} interaction${iids.length === 1 ? "" : "s"}`
      : "no session linked";
  }
  if (!sid) {
    body.innerHTML = `<p class="muted">This trace has no session reference (older / oracle-seeded data).</p>`;
    return;
  }
  body.innerHTML = `<p class="muted">Loading session…</p>`;
  const detail = await fetchSession(sid);
  if (!detail) {
    body.innerHTML = `<p class="muted">Session <code>${escapeHtml(sid)}</code> not found (it may have been deleted).</p>`;
    return;
  }
  const wanted = new Set(iids);
  const matches = (detail.messages || []).filter((m) => wanted.has(m.id));
  const title = escapeHtml(detail.session?.title || "Untitled session");
  const created = detail.session?.created_at || "";
  const headerHtml = `
    <div class="src-head">
      <div><strong>${title}</strong></div>
      <div class="muted small">
        <code>${escapeHtml(sid)}</code>
        ${created ? `· created ${escapeHtml(String(created).slice(0, 19).replace("T", " "))}` : ""}
      </div>
    </div>`;
  let turnsHtml;
  if (!matches.length) {
    turnsHtml = `<p class="muted small">No matching interaction rows found in the raw log
      (it may have been pruned). Linked IDs:
      ${iids.map((i) => `<code>${escapeHtml(i.slice(0, 8))}</code>`).join(", ") || "—"}.</p>`;
  } else {
    turnsHtml = `<ol class="src-turns">${matches
      .map((m, i) => {
        const ts = m.timestamp ? String(m.timestamp).slice(11, 19) : "";
        const q = escapeHtml(m.query || "");
        const r = escapeHtml(m.response || "");
        return `<li>
          <div class="src-turn-head muted small">
            #${i + 1} · <code>${escapeHtml((m.id || "").slice(0, 8))}</code>
            ${ts ? `· ${ts}` : ""}
          </div>
          <div class="src-turn-q"><span class="src-role">user</span> ${q}</div>
          <div class="src-turn-a"><span class="src-role">assistant</span> ${r}</div>
        </li>`;
      })
      .join("")}</ol>`;
  }
  body.innerHTML = headerHtml + turnsHtml;
}

function renderBreakdown(e) {
  const sig = e.signals || {};
  const t = memPolicy?.threshold ?? 0.3;
  const u = sig.uncertainty ?? 0;
  const md = e.metadata || {};
  const extras = (md && md.extras) || {};
  const reason = extras.route_reason || "";
  const accepted = u >= t;

  $("#mem-breakdown-formula").textContent = `score = U`;

  const bar = (label, signal, hint) => {
    const pct = (Math.min(1, Math.max(0, signal)) * 100).toFixed(1);
    return `<div class="bd-row" title="${escapeHtml(hint)}">
      <div class="bd-label"><code>${label}</code></div>
      <div class="bd-bar"><span style="width:${pct}%"></span></div>
      <div class="bd-num">${fmt(signal)}</div>
    </div>`;
  };

  const oracleNote =
    md.extras && md.extras.oracle
      ? `<p class="muted small">Augmented by ${escapeHtml(md.extras.oracle_name || "oracle")} — the response above is the teacher's answer, not the cortex's original output.</p>`
      : "";

  const verdict = accepted
    ? `<span class="badge ok">accepted (U ${fmt(u, 3)} ≥ ${fmt(t)})</span>`
    : `<span class="badge subtle">below threshold (${fmt(u, 3)} < ${fmt(t)})</span>`;

  $("#mem-breakdown-body").innerHTML = `
    ${bar("U", u, "uncertainty: 1 - exp(mean log p) over response tokens")}
    <div class="bd-total">
      <span>gate</span>
      <strong>${fmt(u, 3)}</strong>
      ${verdict}
    </div>
    ${oracleNote}
  `;
  $("#mem-extras-body").innerHTML = reason ? `<div class="bd-reason"><b>Reason:</b> ${escapeHtml(reason)}</div>` : "";
}

function openMemEditor(id) {
  const e = memEntries.find((x) => x.trace_id === id);
  if (!e) return;
  memEditingId = id;
  $("#mem-edit-id").textContent = id;
  $("#mem-query").value = e.query || "";
  $("#mem-response").value = e.response || "";
  renderBreakdown(e);
  renderSource(e);
  const dlg = $("#mem-editor");
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
}

function closeMemEditor() {
  memEditingId = null;
  const dlg = $("#mem-editor");
  if (typeof dlg.close === "function" && dlg.open) dlg.close();
  else dlg.removeAttribute("open");
}

async function saveMem() {
  if (!memEditingId) return;
  try {
    await jpatch(`/api/neocortex/${memEditingId}`, {
      query: $("#mem-query").value,
      response: $("#mem-response").value,
    });
    $("#mem-status").textContent = `saved ${memEditingId}`;
    closeMemEditor();
    await refreshMemory();
  } catch (e) {
    $("#mem-status").textContent = `save failed: ${e.message}`;
  }
}

async function deleteMem(id) {
  const target = id || memEditingId;
  if (!target) return;
  if (!confirm(`Delete trace ${target.slice(0, 8)}…?`)) return;
  try {
    await jdelete(`/api/neocortex/${target}`);
    $("#mem-status").textContent = `deleted ${target}`;
    if (target === memEditingId) closeMemEditor();
    await refreshMemory();
  } catch (e) {
    $("#mem-status").textContent = `delete failed: ${e.message}`;
  }
}

export async function initMemoryTab() {
  $("#mem-refresh").addEventListener("click", refreshMemory);
  $("#mem-search").addEventListener("input", (ev) => {
    memQuery = ev.target.value;
    renderMemRows();
  });
  $("#mem-table tbody").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    const tr = ev.target.closest("tr");
    const id = tr?.dataset.id;
    if (!id) return;
    if (btn.dataset.act === "edit") openMemEditor(id);
    if (btn.dataset.act === "del") deleteMem(id);
  });
  $("#mem-save").addEventListener("click", saveMem);
  $("#mem-delete").addEventListener("click", () => deleteMem());
  $("#mem-cancel").addEventListener("click", closeMemEditor);
  $("#mem-cancel-2").addEventListener("click", closeMemEditor);
  // close on backdrop click (clicks on the <dialog> itself, not its body)
  $("#mem-editor").addEventListener("click", (ev) => {
    if (ev.target.id === "mem-editor") closeMemEditor();
  });
  await refreshMemory();
}
