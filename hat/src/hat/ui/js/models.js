import { $, escapeHtml, toast } from "./util.js";
import { jget, jpost, jdelete } from "./api.js";

let _items = [];      // last fetched catalog items for current backend
let _activeKey = null; // "backend/id" of active model
// Track in-flight SSE downloads keyed by "backend/id" so the card UI can
// render a progress bar + Cancel button and so a Cancel click can find the
// EventSource to close after telling the server to abort.
const _downloads = new Map(); // key -> { es, backend, id, state, ... }

function fmtSize(gb) {
  if (gb == null) return "";
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(gb * 1024).toFixed(0)} MB`;
}

function fmtBytes(n) {
  if (!n || n < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function publisherOf(repo) {
  if (!repo) return "";
  const i = repo.indexOf("/");
  return i > 0 ? repo.slice(0, i) : repo;
}

function statusClass(it) {
  if (_activeKey === `${it.backend}/${it.id}`) return "active";
  if (it.installed) return "installed";
  return "missing";
}

function statusBadge(it) {
  if (_activeKey === `${it.backend}/${it.id}`) {
    return `<span class="badge accent">● Active</span>`;
  }
  if (it.installed) return `<span class="badge ok">✓ Installed</span>`;
  return `<span class="badge subtle">↓ Not installed</span>`;
}

function modelCardHtml(it) {
  const cls = statusClass(it);
  const key = `${it.backend}/${it.id}`;
  const dl = _downloads.get(key);
  const meta = [publisherOf(it.repo_id), fmtSize(it.size_gb)]
    .filter(Boolean)
    .join(" · ");

  let footer;
  if (dl) {
    // In-flight download: render a progress bar + Cancel button instead
    // of the regular action row. The actual values are kept in sync by
    // ``renderDownloadProgress`` on each SSE tick so we don't re-render
    // the whole grid 10× a second.
    const pct = dl.bytesTotal > 0
      ? Math.min(100, (dl.bytesDone / dl.bytesTotal) * 100)
      : 0;
    footer = `
      <div class="dl-progress" data-key="${escapeHtml(key)}">
        <div class="dl-bar"><div class="dl-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <div class="dl-meta mono">
          <span class="dl-pct">${pct.toFixed(1)}%</span>
          <span class="dl-bytes">${fmtBytes(dl.bytesDone)} / ${fmtBytes(dl.bytesTotal)}</span>
          <span class="dl-files">${dl.filesDone}/${dl.filesTotal} files</span>
        </div>
      </div>
      <footer class="row gap wrap">
        <button class="ghost small danger" data-act="cancel">Cancel</button>
      </footer>`;
  } else if (it.installed) {
    footer = `<footer class="row gap wrap">
      <button class="primary small" data-act="use">Use</button>
      <button class="ghost small danger" data-act="delete">Delete</button>
    </footer>`;
  } else {
    footer = `<footer class="row gap wrap">
      <button class="primary small" data-act="download">Download</button>
    </footer>`;
  }

  return `
    <article class="model-card ${cls}" data-id="${escapeHtml(it.id)}">
      <header>
        <div class="model-title" title="${escapeHtml(it.repo_id)}">
          ${escapeHtml(it.display)}
        </div>
        ${statusBadge(it)}
      </header>
      <div class="model-meta">${escapeHtml(meta || "—")}</div>
      ${it.notes
        ? `<p class="model-notes">${escapeHtml(it.notes)}</p>`
        : ""}
      <div class="model-id mono">${escapeHtml(it.repo_id)}</div>
      ${footer}
    </article>`;
}

// Cheap, targeted DOM update for SSE progress ticks. Re-rendering the
// whole grid would flicker selection state and is unnecessary.
function renderDownloadProgress(key) {
  const dl = _downloads.get(key);
  if (!dl) return;
  const node = document.querySelector(`.dl-progress[data-key="${CSS.escape(key)}"]`);
  if (!node) return;
  const pct = dl.bytesTotal > 0
    ? Math.min(100, (dl.bytesDone / dl.bytesTotal) * 100)
    : 0;
  node.querySelector(".dl-fill").style.width = `${pct.toFixed(1)}%`;
  node.querySelector(".dl-pct").textContent = `${pct.toFixed(1)}%`;
  node.querySelector(".dl-bytes").textContent =
    `${fmtBytes(dl.bytesDone)} / ${fmtBytes(dl.bytesTotal)}`;
  node.querySelector(".dl-files").textContent =
    `${dl.filesDone}/${dl.filesTotal} files`;
}

function applyFilters() {
  const q = ($("#mgr-search")?.value || "").trim().toLowerCase();
  const f = $("#mgr-filter")?.value || "all";
  const filtered = _items.filter((it) => {
    if (f === "installed" && !it.installed) return false;
    if (f === "missing" && it.installed) return false;
    if (!q) return true;
    return (
      it.id.toLowerCase().includes(q) ||
      it.repo_id.toLowerCase().includes(q) ||
      (it.display || "").toLowerCase().includes(q) ||
      (it.notes || "").toLowerCase().includes(q)
    );
  });

  const grid = $("#model-grid");
  if (!grid) return;
  grid.innerHTML = filtered.length
    ? filtered.map(modelCardHtml).join("")
    : `<div class="empty">No models match the current filter.</div>`;

  const installed = _items.filter((i) => i.installed).length;
  $("#models-count-installed").textContent = `${installed} installed`;
  $("#models-count-total").textContent = `${_items.length} total`;
}

export async function loadActive() {
  try {
    const a = await jget("/api/models/active");
    _activeKey = a ? `${a.backend}/${a.id}` : null;
    $("#active-model").textContent = a
      ? `Active: ${a.backend} / ${a.id}`
      : "Active: —";
    return a;
  } catch {
    _activeKey = null;
    $("#active-model").textContent = "Active: —";
    return null;
  }
}

export async function loadCatalog(backend) {
  try {
    const data = await jget(`/api/models?backend=${encodeURIComponent(backend)}`);
    _items = (data.items || []).map((i) => ({ ...i, backend }));

    // Always know which model is active so the dropdowns can reflect it.
    await loadActive();

    const cm = $("#chat-model");
    if (cm) {
      const installed = _items.filter((i) => i.installed);
      cm.innerHTML = installed.length
        ? installed
            .map((i) => `<option value="${i.id}">${escapeHtml(i.display)}</option>`)
            .join("")
        : `<option value="">(none installed)</option>`;
      // If the active model belongs to this backend and is installed, pin it.
      if (_activeKey && _activeKey.startsWith(`${backend}/`)) {
        const activeId = _activeKey.slice(backend.length + 1);
        if (installed.some((i) => i.id === activeId)) {
          cm.value = activeId;
        }
      }
    }

    if ($("#model-grid")) applyFilters();
  } catch (e) {
    const status = $("#mgr-status");
    if (status) status.textContent = `load failed: ${e.message}`;
  }
}

async function downloadModelById(backend, id) {
  const key = `${backend}/${id}`;
  if (_downloads.has(key)) {
    toast(`Download already running for ${key}`, "info");
    return;
  }

  const url = `/api/models/download/stream?backend=${encodeURIComponent(backend)}&id=${encodeURIComponent(id)}`;
  const es = new EventSource(url);
  const state = {
    es, backend, id,
    bytesDone: 0, bytesTotal: 0,
    filesDone: 0, filesTotal: 0,
    cancelling: false,
  };
  _downloads.set(key, state);

  $("#mgr-status").textContent = `downloading ${key}…`;
  applyFilters(); // swap the card footer to the progress bar

  const onStart = (e) => {
    const d = JSON.parse(e.data);
    state.bytesTotal = d.bytes_total || 0;
    state.filesTotal = d.files_total || 0;
    renderDownloadProgress(key);
  };
  const onProgress = (e) => {
    const d = JSON.parse(e.data);
    state.bytesDone = d.bytes_done ?? state.bytesDone;
    state.bytesTotal = d.bytes_total ?? state.bytesTotal;
    state.filesDone = d.files_done ?? state.filesDone;
    state.filesTotal = d.files_total ?? state.filesTotal;
    renderDownloadProgress(key);
  };
  const finish = (kind, msg) => {
    es.close();
    _downloads.delete(key);
    const status = $("#mgr-status");
    if (kind === "done") {
      if (status) status.textContent = `downloaded ${key}`;
      toast(`Downloaded ${key}`, "ok");
    } else if (kind === "cancelled") {
      if (status) status.textContent = `cancelled ${key}`;
      toast(`Cancelled ${key}`, "info");
    } else {
      if (status) status.textContent = `download failed: ${msg}`;
      toast(`Download failed: ${msg}`, "error");
    }
    loadCatalog(backend); // refresh installed/missing state + footer
  };

  es.addEventListener("start", onStart);
  es.addEventListener("progress", onProgress);
  es.addEventListener("done", () => finish("done"));
  es.addEventListener("cancelled", () => finish("cancelled"));
  es.addEventListener("error", (e) => {
    // ``error`` events from the server carry a payload; transport-level
    // errors (network drop, server gone) arrive as the default error
    // event with no ``data``. Distinguish so we don't double-report.
    if (e.data) {
      let msg = "stream error";
      try { msg = JSON.parse(e.data).message || msg; } catch { /* ignore */ }
      finish("error", msg);
    } else if (es.readyState === EventSource.CLOSED) {
      finish("error", "connection closed");
    }
  });
}

async function cancelDownloadById(backend, id) {
  const key = `${backend}/${id}`;
  const dl = _downloads.get(key);
  if (!dl || dl.cancelling) return;
  dl.cancelling = true;
  $("#mgr-status").textContent = `cancelling ${key}…`;
  try {
    await jpost("/api/models/download/cancel", { backend, id });
    // The server will emit a ``cancelled`` SSE event and close the
    // stream, which triggers cleanup via the listener above.
  } catch (e) {
    dl.cancelling = false;
    toast(`Cancel failed: ${e.message}`, "error");
  }
}

async function deleteModelById(backend, id) {
  if (!confirm(
    `Delete ${backend}/${id} from disk?\n\n` +
    `This removes the local weights but the catalog entry remains, ` +
    `so you can re-download it later.`,
  )) return;
  $("#mgr-status").textContent = `deleting ${backend}/${id}…`;
  try {
    await jdelete(`/api/models/${encodeURIComponent(backend)}/${encodeURIComponent(id)}`);
    $("#mgr-status").textContent = `deleted ${backend}/${id}`;
    toast(`Deleted ${backend}/${id}`, "ok");
    await loadCatalog(backend);
  } catch (e) {
    $("#mgr-status").textContent = `delete failed: ${e.message}`;
    toast(`Delete failed: ${e.message}`, "error");
  }
}

export async function activateModel(backend, id) {
  const status = $("#mgr-status");
  if (status) status.textContent = `loading ${backend}/${id}…`;
  // Surface a non-blocking "loading" toast so the user knows something is
  // happening; some backends (large HF checkpoints) take 10-30s to load.
  const dismissLoading = toast(`Loading ${backend}/${id}…`, "info", { timeout: 0 });
  try {
    await jpost("/api/models/active", { backend, id });
    if (status) status.textContent = `active: ${backend}/${id}`;
    dismissLoading();
    toast(`Active model: ${backend}/${id}`, "ok");
    await loadActive();
    await loadCatalog(backend);
  } catch (e) {
    if (status) status.textContent = `activate failed: ${e.message}`;
    dismissLoading();
    // Sticky error toast — the message contains the underlying exception
    // (OOM, missing extras, bad path, …) and the user needs time to read it.
    toast(`Activate failed: ${e.message}`, "error");
  }
}

export async function unloadActive() {
  const status = $("#mgr-status");
  try {
    await jdelete("/api/models/active");
    if (status) status.textContent = "all models unloaded";
    toast("All models unloaded", "ok");
    await loadActive();
    if ($("#model-grid")) applyFilters();
  } catch (e) {
    if (status) status.textContent = `unload failed: ${e.message}`;
    toast(`Unload failed: ${e.message}`, "error");
  }
}

export async function downloadModel() {
  // Legacy entry point kept for backwards-compat callers.
  const backend = $("#mgr-backend").value;
  const target = _items.find((i) => !i.installed);
  if (!target) {
    $("#mgr-status").textContent = "nothing to download (all installed)";
    return;
  }
  await downloadModelById(backend, target.id);
}

export async function initModelsTab() {
  const backendSel = $("#mgr-backend");
  // Resolve the default backend from the server: prefer whichever backend
  // hosts the currently-active model; otherwise honour HAT_CORTEX_BACKEND.
  const [active, health] = await Promise.all([
    loadActive(),
    jget("/healthz").catch(() => null),
  ]);
  const wanted =
    (active && active.backend) ||
    (health && health.cortex_backend && health.cortex_backend !== "noop"
      ? health.cortex_backend
      : null);
  if (wanted && [...backendSel.options].some((o) => o.value === wanted)) {
    backendSel.value = wanted;
  }
  backendSel.addEventListener("change", () => loadCatalog(backendSel.value));
  $("#mgr-filter").addEventListener("change", applyFilters);
  $("#mgr-search").addEventListener("input", applyFilters);
  $("#mgr-refresh").addEventListener("click", () => loadCatalog(backendSel.value));

  $("#model-grid").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const card = btn.closest(".model-card");
    const id = card?.dataset.id;
    if (!id) return;
    const backend = backendSel.value;
    const act = btn.dataset.act;
    if (act === "use") activateModel(backend, id);
    else if (act === "download") downloadModelById(backend, id);
    else if (act === "cancel") cancelDownloadById(backend, id);
    else if (act === "delete") deleteModelById(backend, id);
  });

  await loadCatalog(backendSel.value);
}
