import { $, escapeHtml } from "./util.js";
import { jget, jpatch, jdelete } from "./api.js";

let memEntries = [];
let memEditingId = null;
let memQuery = "";

function memRowHtml(e, i) {
  const score = (e.score ?? 0).toFixed(2);
  const q = escapeHtml((e.query || "").slice(0, 100));
  const r = escapeHtml((e.response || "").slice(0, 200));
  return `<tr data-id="${e.trace_id}">
    <td>${i + 1}</td>
    <td><code>${escapeHtml((e.trace_id || "").slice(0, 8))}</code></td>
    <td>${score}</td>
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
    tbody.innerHTML = `<tr><td colspan="6" class="empty">No entries match.</td></tr>`;
  }
  $("#mem-count").textContent = `${filtered.length} / ${memEntries.length}`;
}

export async function refreshMemory() {
  $("#mem-status").textContent = "loading…";
  try {
    const data = await jget("/api/neocortex");
    memEntries = data.data || [];
    renderMemRows();
    $("#mem-status").textContent = `${memEntries.length} entries`;
  } catch (e) {
    $("#mem-status").textContent = `load failed: ${e.message}`;
  }
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
