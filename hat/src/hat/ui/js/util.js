export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// --- toast notifications --------------------------------------------------
// Lightweight, dependency-free notifier. We use it to surface async errors
// (model activation, downloads, etc.) that would otherwise only land in a
// status-bar text and be missed by the user. A single fixed-position stack
// is created lazily on first call.
//
// `kind` ∈ {"info", "ok", "error", "warn"} drives the accent color.
// Errors are sticky by default (timeout=0) so the user has time to read
// them; success/info auto-dismiss after ~4s.
export function toast(message, kind = "info", { timeout } = {}) {
  let stack = document.getElementById("toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.id = "toast-stack";
    stack.className = "toast-stack";
    document.body.appendChild(stack);
  }
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.innerHTML =
    `<span class="toast-msg"></span>` +
    `<button class="toast-close" aria-label="dismiss">×</button>`;
  el.querySelector(".toast-msg").textContent = String(message ?? "");
  const close = () => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 180);
  };
  el.querySelector(".toast-close").addEventListener("click", close);
  stack.appendChild(el);
  const ttl = timeout ?? (kind === "error" ? 0 : 4000);
  if (ttl > 0) setTimeout(close, ttl);
  return close;
}

// Split a chunk of streaming text into <think>…</think> blocks and the rest.
// showThink=true keeps the block as a styled aside; false drops it.
export function renderThink(text, showThink) {
  const out = [];
  const re = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: "text", value: text.slice(last, m.index) });
    if (showThink) out.push({ kind: "think", value: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", value: text.slice(last) });
  return out;
}

export function renderBubbleHtml(content, { showThink = false } = {}) {
  return renderThink(content || "", showThink)
    .map((p) =>
      p.kind === "think"
        ? `<span class="think">${escapeHtml(p.value)}</span>`
        : escapeHtml(p.value),
    )
    .join("");
}
