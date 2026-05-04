import { $, escapeHtml } from "./util.js";
import { jget, jpost, jdelete } from "./api.js";

let _items = [];      // last fetched catalog items for current backend
let _activeKey = null; // "backend/id" of active model

function fmtSize(gb) {
  if (gb == null) return "";
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(gb * 1024).toFixed(0)} MB`;
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
  const meta = [publisherOf(it.repo_id), fmtSize(it.size_gb)]
    .filter(Boolean)
    .join(" · ");
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
      <footer class="row gap wrap">
        ${it.installed
          ? `<button class="primary small" data-act="use">Use</button>
             <button class="ghost small danger" data-act="delete">Delete</button>`
          : `<button class="primary small" data-act="download">Download</button>`}
      </footer>
    </article>`;
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
  $("#mgr-status").textContent = `downloading ${backend}/${id}…`;
  try {
    const r = await jpost("/api/models/download", { backend, id });
    $("#mgr-status").textContent = `downloaded → ${r.local_dir}`;
    await loadCatalog(backend);
  } catch (e) {
    $("#mgr-status").textContent = `download failed: ${e.message}`;
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
    await loadCatalog(backend);
  } catch (e) {
    $("#mgr-status").textContent = `delete failed: ${e.message}`;
  }
}

export async function activateModel(backend, id) {
  const status = $("#mgr-status");
  if (status) status.textContent = `loading ${backend}/${id}…`;
  try {
    await jpost("/api/models/active", { backend, id });
    if (status) status.textContent = `active: ${backend}/${id}`;
    await loadActive();
    await loadCatalog(backend);
  } catch (e) {
    if (status) status.textContent = `activate failed: ${e.message}`;
  }
}

export async function unloadActive() {
  const status = $("#mgr-status");
  try {
    await jdelete("/api/models/active");
    if (status) status.textContent = "all models unloaded";
    await loadActive();
    if ($("#model-grid")) applyFilters();
  } catch (e) {
    if (status) status.textContent = `unload failed: ${e.message}`;
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
    else if (act === "delete") deleteModelById(backend, id);
  });

  await loadCatalog(backendSel.value);
}
