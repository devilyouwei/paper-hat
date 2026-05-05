import { $, escapeHtml } from "./util.js";
import { jget, jpatch, jdelete } from "./api.js";

let memEntries = [];
let memEditingId = null;
let memQuery = "";
// Live write-policy coefficients (α, β, γ, threshold). Loaded from
// /api/policy on first refresh and reused for the breakdown view.
let memPolicy = null;

function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function memRowHtml(e, i) {
  const score = fmt(e.score);
  const sig = e.signals || {};
  const u = fmt(sig.uncertainty);
  const f = fmt(sig.feedback);
  const n = fmt(sig.novelty);
  const q = escapeHtml((e.query || "").slice(0, 100));
  const r = escapeHtml((e.response || "").slice(0, 200));
  // Oracle-augmented traces carry ``metadata.extras.oracle = true`` and
  // their ``metadata.source`` ends with ``+oracle``. Show a small badge
  // so users can distinguish teacher-corrected examples from human or
  // self-supervised ones at a glance.
  const md = e.metadata || {};
  const isOracle =
    (md.extras && md.extras.oracle) ||
    (typeof md.source === "string" && md.source.includes("oracle"));
  const oracleBadge = isOracle
    ? `<span class="badge oracle" title="${escapeHtml(
        (md.extras && md.extras.oracle_name) || "oracle",
      )}">oracle</span>`
    : "";
  // Highlight which of α·U / β·F / γ·N dominated this entry's score.
  // The signal cell with the largest weighted contribution is bolded so
  // the user can see at a glance why a trace was kept.
  const a = memPolicy?.alpha ?? 0.4;
  const b = memPolicy?.beta ?? 0.4;
  const g = memPolicy?.gamma ?? 0.2;
  const contribs = {
    uncertainty: a * (sig.uncertainty ?? 0),
    feedback: b * (sig.feedback ?? 0),
    novelty: g * (sig.novelty ?? 0),
  };
  const dominant = Object.entries(contribs).sort((a, b) => b[1] - a[1])[0][0];
  const cell = (k, v) =>
    `<td class="${dominant === k ? "dominant" : ""}">${v}</td>`;
  return `<tr data-id="${e.trace_id}">
    <td>${i + 1}</td>
    <td><code>${escapeHtml((e.trace_id || "").slice(0, 8))}</code>${oracleBadge}</td>
    <td><strong>${score}</strong></td>
    ${cell("uncertainty", u)}
    ${cell("feedback", f)}
    ${cell("novelty", n)}
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
    tbody.innerHTML = `<tr><td colspan="9" class="empty">No entries match.</td></tr>`;
  }
  $("#mem-count").textContent = `${filtered.length} / ${memEntries.length}`;
}

function renderPolicy() {
  if (!memPolicy) return;
  const { alpha, beta, gamma, threshold } = memPolicy;
  const a = fmt(alpha);
  const b = fmt(beta);
  const g = fmt(gamma);
  const t = fmt(threshold);
  $("#mem-policy-summary").textContent =
    `score = ${a}·U + ${b}·F + ${g}·N · threshold ${t}`;
  $("#mem-policy-alpha").textContent = a;
  $("#mem-policy-beta").textContent = b;
  $("#mem-policy-gamma").textContent = g;
  $("#mem-formula").textContent =
    `score(m) = ${a}·U(m) + ${b}·F(m) + ${g}·N(m)    ·    write if score > ${t}`;
  if (memPolicy.oracle) {
    const o = memPolicy.oracle;
    if (o.enabled) {
      $("#mem-oracle-summary").textContent =
        `Oracle: ${o.model} consulted when U > ${fmt(o.threshold)} and the user has not corrected the response. Limits: ${fmt(o.rps, 2)}/s, ${o.daily_calls}/day.`;
    } else {
      $("#mem-oracle-summary").textContent =
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

function renderBreakdown(e) {
  const sig = e.signals || {};
  const a = memPolicy?.alpha ?? 0.4;
  const b = memPolicy?.beta ?? 0.4;
  const g = memPolicy?.gamma ?? 0.2;
  const t = memPolicy?.threshold ?? 0.3;
  const u = sig.uncertainty ?? 0;
  const f = sig.feedback ?? 0;
  const n = sig.novelty ?? 0;
  const cu = a * u;
  const cf = b * f;
  const cn = g * n;
  const total = cu + cf + cn;
  const forced = f >= 1.0;
  const accepted = forced || total > t;

  $("#mem-breakdown-formula").textContent =
    `score = ${fmt(a)}·U + ${fmt(b)}·F + ${fmt(g)}·N`;

  // Bar widths are scaled to the largest single contribution so the
  // visual comparison is honest even when the score is tiny.
  const maxC = Math.max(cu, cf, cn, 0.001);
  const bar = (label, weight, signal, contrib, hint) => {
    const pct = ((contrib / maxC) * 100).toFixed(1);
    return `<div class="bd-row" title="${escapeHtml(hint)}">
      <div class="bd-label"><code>${label}</code> <span class="muted small">×${fmt(weight)}</span></div>
      <div class="bd-bar"><span style="width:${pct}%"></span></div>
      <div class="bd-num">${fmt(signal)}</div>
      <div class="bd-num bd-contrib">+${fmt(contrib, 3)}</div>
    </div>`;
  };

  const md = e.metadata || {};
  const oracleNote =
    md.extras && md.extras.oracle
      ? `<p class="muted small">Augmented by ${escapeHtml(md.extras.oracle_name || "oracle")} — the response above is the teacher's correction, not the cortex's original answer.</p>`
      : "";

  const verdict = accepted
    ? forced
      ? `<span class="badge ok">accepted (forced: F = 1)</span>`
      : `<span class="badge ok">accepted (score ${fmt(total, 3)} > ${fmt(t)})</span>`
    : `<span class="badge subtle">below threshold (${fmt(total, 3)} ≤ ${fmt(t)})</span>`;

  $("#mem-breakdown-body").innerHTML = `
    ${bar("U", a, u, cu, "uncertainty: 1 - exp(mean log p) over response tokens")}
    ${bar("F", b, f, cf, "feedback: 1.0 with correction, else explicit feedback or LLM judge")}
    ${bar("N", g, n, cn, "novelty: how new the user input is to the model")}
    <div class="bd-total">
      <span>total</span>
      <strong>${fmt(total, 3)}</strong>
      ${verdict}
    </div>
    ${oracleNote}
  `;
}

function openMemEditor(id) {
  const e = memEntries.find((x) => x.trace_id === id);
  if (!e) return;
  memEditingId = id;
  $("#mem-edit-id").textContent = id;
  $("#mem-query").value = e.query || "";
  $("#mem-response").value = e.response || "";
  $("#mem-score").value = e.score ?? 0;
  $("#mem-score-out").textContent = (e.score ?? 0).toFixed(2);
  renderBreakdown(e);
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
      score: parseFloat($("#mem-score").value),
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
  $("#mem-score").addEventListener("input", (e) => {
    $("#mem-score-out").textContent = parseFloat(e.target.value).toFixed(2);
  });
  await refreshMemory();
}
