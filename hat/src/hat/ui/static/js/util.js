export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
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
