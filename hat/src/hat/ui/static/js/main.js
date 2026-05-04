import { $, $$ } from "./util.js";
import { jget } from "./api.js";
import { loadActive } from "./models.js";

// ---------- partial loader (lazy) --------------------------------------

const initializers = {
  chat: () => import("./chat.js").then((m) => m.initChatTab()),
  models: () => import("./models.js").then((m) => m.initModelsTab()),
  memory: () => import("./memory.js").then((m) => m.initMemoryTab()),
};

const loaded = new Set();

async function ensureTab(name) {
  if (loaded.has(name)) return;
  const host = $(`#tab-${name}`);
  if (!host) return;
  const url = `/ui/static/partials/${name}.html`;
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) {
    host.innerHTML = `<p class="status">failed to load ${name}: ${r.status}</p>`;
    return;
  }
  host.innerHTML = await r.text();
  loaded.add(name);
  try {
    await initializers[name]?.();
  } catch (e) {
    host.insertAdjacentHTML(
      "afterbegin",
      `<p class="status">init failed: ${e.message}</p>`,
    );
  }
}

function activateTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  ensureTab(name);
  // Memory tab is a snapshot — refresh whenever the tab is reopened.
  if (name === "memory" && loaded.has("memory")) {
    import("./memory.js").then((m) => m.refreshMemory());
  }
}

$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});

// ---------- topbar status ----------------------------------------------

async function loadHealth() {
  const dot = $("#health-dot");
  try {
    const h = await jget("/healthz");
    dot.classList.add("ok");
    dot.title = `backend: ${h.cortex_backend}`;
    // Stash the env-configured backend so tab modules can default their
    // backend selectors to it on first load. ``noop`` is the inert
    // fallback when no real cortex is configured — we treat it as "no
    // preference" (the dropdown's first option wins).
    if (h.cortex_backend && h.cortex_backend !== "noop") {
      window.__hatEnvBackend = h.cortex_backend;
    }
  } catch {
    dot.classList.add("err");
    dot.title = "server unreachable";
  }
}

// ---------- boot --------------------------------------------------------

(async function boot() {
  await loadHealth();
  await loadActive();
  // First tab: render Chat immediately. Others lazy-load on click.
  await ensureTab("chat");
})();
