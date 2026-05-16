import { $, $$ } from "./util.js";
import { jget } from "./api.js";
import { loadActive } from "./models.js";

// ---------- boot-time fetches cache ------------------------------------
// Other tabs (chat.js, models.js) need the same /healthz and active-model
// info we already fetched during boot. Expose accessors so they don't issue
// duplicate requests on page load.
let _bootHealth = null;
let _bootActive = null;

export function getBootHealth() {
  return _bootHealth;
}
export function getBootActive() {
  return _bootActive;
}

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
  const url = `/ui/partials/${name}.html`;
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
    _bootHealth = h;
    dot.classList.add("ok");
    dot.title = `backend: ${h.cortex_backend}`;
  } catch {
    dot.classList.add("err");
    dot.title = "server unreachable";
  }
}

// ---------- theme toggle -----------------------------------------------

function currentEffectiveTheme() {
  return document.documentElement.dataset.effectiveTheme || "dark";
}

function applyTheme(next) {
  const html = document.documentElement;
  if (next) {
    html.setAttribute("data-theme", next);
    html.dataset.effectiveTheme = next;
    try { localStorage.setItem("hat-theme", next); } catch (e) {}
  } else {
    // "system" — clear override and re-derive from OS preference.
    html.removeAttribute("data-theme");
    const prefersLight =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: light)").matches;
    html.dataset.effectiveTheme = prefersLight ? "light" : "dark";
    try { localStorage.removeItem("hat-theme"); } catch (e) {}
  }
}

function initThemeToggle() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    applyTheme(currentEffectiveTheme() === "light" ? "dark" : "light");
  });
  // If the user hasn't set an explicit override, follow the OS preference live.
  if (window.matchMedia) {
    const mql = window.matchMedia("(prefers-color-scheme: light)");
    mql.addEventListener?.("change", (ev) => {
      try {
        if (localStorage.getItem("hat-theme")) return; // user override wins
      } catch (e) {}
      document.documentElement.dataset.effectiveTheme = ev.matches
        ? "light"
        : "dark";
    });
  }
}

// ---------- boot --------------------------------------------------------

(async function boot() {
  initThemeToggle();
  await loadHealth();
  _bootActive = await loadActive();
  // First tab: render Chat immediately. Others lazy-load on click.
  await ensureTab("chat");
})();
