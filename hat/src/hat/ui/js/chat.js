import { $, $$, escapeHtml, renderBubbleHtml } from "./util.js";
import { jget, jpost, jdelete } from "./api.js";
import { loadCatalog, loadActive } from "./models.js";
import {
  appendTraceEvent,
  clearTracePanel,
  initTracePanel,
  loadTracesForSession,
} from "./traces.js";

let currentSessionId = null;
// In-memory transcript for the active session, mirroring what the backend
// has on disk. Sent verbatim with every /v1/chat/completions request so the
// model sees prior turns as context.
let history = [];

function chatbox() {
  return $("#chatbox");
}

/* --- smart auto-scroll ---------------------------------------------------
 * Stick to the bottom while streaming, but release the lock as soon as the
 * user scrolls up. Re-engage the lock when the user scrolls back near the
 * bottom. */
let stickToBottom = true;
const NEAR_BOTTOM_PX = 60;

function isNearBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM_PX;
}

function scrollToBottom(force = false) {
  const el = chatbox();
  if (!el) return;
  if (force || stickToBottom) {
    el.scrollTop = el.scrollHeight;
  }
}

function installScrollWatcher() {
  const el = chatbox();
  if (!el || el.dataset.scrollWatcher) return;
  el.dataset.scrollWatcher = "1";
  el.addEventListener(
    "scroll",
    () => {
      stickToBottom = isNearBottom(el);
    },
    { passive: true },
  );
}

function appendBubble(role, content) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.innerHTML = renderBubbleHtml(content, {
    showThink: $("#show-thinking")?.checked,
  });
  chatbox().appendChild(div);
  // A new bubble means a new turn — always pin to bottom and re-arm the lock.
  stickToBottom = true;
  scrollToBottom(true);
  return div;
}

/* Show the cortex's uncertainty on the assistant bubble as a small badge.
 * The ``uncertainty`` event fires first and carries the raw U value; the
 * ``skipped`` (gate rejected) and ``dropped`` (router judged not worth
 * remembering) events update the badge with a status suffix. */
function attachUncertaintyBadge(bubble, ev) {
  if (!bubble || !ev) return;
  let badge = bubble.querySelector(".uncertainty-badge");
  const u = ev.uncertainty ?? ev.signals?.uncertainty;
  if (!badge) {
    if (typeof u !== "number") return;
    badge = document.createElement("span");
    badge.className = "uncertainty-badge";
    bubble.appendChild(badge);
  }
  const current = badge.dataset.u ? Number(badge.dataset.u) : u;
  if (typeof u === "number") badge.dataset.u = String(u);
  const stage = ev.stage;
  let status = "";
  if (stage === "skipped") status = " · skipped";
  else if (stage === "dropped") status = " · dropped";
  else if (stage === "routed" && ev.decision) status = ` · ${ev.decision.toLowerCase()}`;
  const shown = typeof current === "number" ? `U=${current.toFixed(2)}` : "U=—";
  badge.textContent = `${shown}${status}`;
  badge.classList.toggle("skipped", stage === "skipped");
  badge.classList.toggle("dropped", stage === "dropped");
  if (ev.reason) badge.title = ev.reason;
}

function clearChat() {
  chatbox().innerHTML = "";
}

async function loadSessions(selectId = null) {
  const data = await jget("/api/sessions");
  const list = $("#session-list");
  list.innerHTML = "";
  for (const s of data.data) {
    const li = document.createElement("li");
    if (s.id === (selectId || currentSessionId)) li.classList.add("active");
    li.dataset.id = s.id;
    li.innerHTML = `
      <span class="title">${escapeHtml(s.title || "New chat")}</span>
      <span class="count">${s.message_count}</span>`;
    li.addEventListener("click", () => openSession(s.id));
    list.appendChild(li);
  }
}

async function openSession(id) {
  currentSessionId = id;
  $("#session-status").textContent = "";
  const detail = await jget(`/api/sessions/${id}`);
  clearChat();
  history = [];
  for (const it of detail.messages) {
    if (it.query) {
      appendBubble("user", it.query);
      history.push({ role: "user", content: it.query });
    }
    if (it.response) {
      const aBubble = appendBubble("assistant", it.response);
      history.push({ role: "assistant", content: it.response });
      // Re-attach the uncertainty / route badge from the persisted record so
      // the user sees the same annotation after a page refresh.
      if (it.hat) {
        const decision = it.hat.decision;
        // Map the persisted decision back to a synthetic event so the badge
        // logic renders the same suffix used during live streaming:
        //   created/revised -> "routed" stage with CREATE/REVISE label
        //   skipped/dropped -> matching terminal stage
        const isAccepted = decision === "created" || decision === "revised";
        const synthetic = isAccepted
          ? {
              stage: "routed",
              uncertainty: it.hat.uncertainty,
              decision: decision === "revised" ? "REVISE" : "CREATE",
              reason: it.hat.reason,
            }
          : {
              stage: decision || undefined,
              uncertainty: it.hat.uncertainty,
              reason: it.hat.reason,
            };
        attachUncertaintyBadge(aBubble, synthetic);
      }
    }
  }
  $$("#session-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.id === id),
  );
  // Reload trace panel with traces already stored for this session.
  await loadTracesForSession(id);
}

async function newSession() {
  try {
    const created = await jpost("/api/sessions", {});
    currentSessionId = created.id;
    history = [];
    clearChat();
    clearTracePanel();
    await loadSessions(created.id);
  } catch (e) {
    $("#session-status").textContent = `new chat failed: ${e.message}`;
  }
}

async function deleteCurrentSession() {
  if (!currentSessionId) {
    $("#session-status").textContent = "no session selected";
    return;
  }
  try {
    await jdelete(`/api/sessions/${currentSessionId}`);
    currentSessionId = null;
    history = [];
    clearChat();
    await initSessions();
  } catch (e) {
    $("#session-status").textContent = `delete failed: ${e.message}`;
  }
}

async function initSessions() {
  const data = await jget("/api/sessions");
  if (!data.data.length) {
    await newSession();
    return;
  }
  await openSession(data.data[0].id);
  await loadSessions();
}

async function sendChat(ev) {
  ev.preventDefault();
  const text = $("#msg").value.trim();
  if (!text) return;
  $("#msg").value = "";
  $("#send-btn").disabled = true;

  appendBubble("user", text);
  const assistantDiv = appendBubble("assistant", "");
  let buffer = "";

  // Build the full message list: prior turns + this user turn. The backend
  // forwards the whole array to the cortex's chat template, which is how
  // the model sees conversational context.
  const outgoing = history.concat([{ role: "user", content: text }]);

  const body = {
    model: "hat-cortex",
    messages: outgoing,
    stream: true,
    temperature: parseFloat($("#temp").value),
    max_tokens: parseInt($("#max-tokens").value, 10),
    chat_template_kwargs: { enable_thinking: $("#enable-thinking").checked },
    session_id: currentSessionId,
  };

  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      assistantDiv.textContent = `error: ${resp.status} ${await resp.text()}`;
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    const showThink = $("#show-thinking").checked;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = pending.indexOf("\n")) >= 0) {
        const line = pending.slice(0, idx).trim();
        pending = pending.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const obj = JSON.parse(payload);
          if (obj.hat_session_id && !currentSessionId) {
            currentSessionId = obj.hat_session_id;
          }
          if (obj.hat_trace_event) {
            attachUncertaintyBadge(assistantDiv, obj.hat_trace_event);
            appendTraceEvent(obj.hat_trace_event);
          }
          const delta = obj.choices?.[0]?.delta?.content || "";
          if (delta) {
            buffer += delta;
            assistantDiv.innerHTML = renderBubbleHtml(buffer, { showThink });
            scrollToBottom(); // only scrolls if user is still near bottom
          }
        } catch {
          // ignore non-JSON keepalives
        }
      }
    }
    // Commit this turn to the in-memory history so the next request sees it.
    history.push({ role: "user", content: text });
    if (buffer) history.push({ role: "assistant", content: buffer });
    await loadSessions(currentSessionId);
  } catch (e) {
    assistantDiv.textContent = `network error: ${e.message}`;
  } finally {
    $("#send-btn").disabled = false;
  }
}

initTracePanel();
export async function initChatTab() {
  // Wiring inside the chat partial
  installScrollWatcher();
  $("#chat-form").addEventListener("submit", sendChat);
  $("#new-chat-btn").addEventListener("click", newSession);
  $("#delete-session-btn").addEventListener("click", deleteCurrentSession);

  $("#chat-backend").addEventListener("change", () =>
    loadCatalog($("#chat-backend").value),
  );
  $("#chat-use-btn").addEventListener("click", async () => {
    const backend = $("#chat-backend").value;
    const id = $("#chat-model").value;
    if (!id) return;
    const { activateModel } = await import("./models.js");
    activateModel(backend, id);
  });
  $("#chat-unload-btn").addEventListener("click", async () => {
    const { unloadActive } = await import("./models.js");
    unloadActive();
  });

  // Enter sends, Shift+Enter inserts newline.
  // Skip while an IME composition is in progress (e.g. Chinese/Japanese input)
  // so that committing a candidate with Enter does not submit the form.
  $("#msg").addEventListener("keydown", (e) => {
    if (
      e.key === "Enter" &&
      !e.shiftKey &&
      !e.isComposing &&
      e.keyCode !== 229
    ) {
      e.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });

  // Inline generation-settings summary: shows current temp / max-tokens
  // when the panel is collapsed so the user does not need to expand it.
  const updateGenSummary = () => {
    const el = $("#gen-summary");
    if (!el) return;
    const t = $("#temp")?.value;
    const m = $("#max-tokens")?.value;
    const think = $("#enable-thinking")?.checked ? " · think" : "";
    el.textContent = `T=${t} · max=${m}${think}`;
  };
  ["#temp", "#max-tokens", "#enable-thinking"].forEach((sel) => {
    const el = $(sel);
    if (el) el.addEventListener("change", updateGenSummary);
  });
  updateGenSummary();

  // Resolve the backend to display *before* loading the catalog. Priority:
  //   1. backend hosting the currently-active model (post-refresh state).
  //   2. ``HAT_CORTEX_BACKEND`` from the server env (via /healthz).
  //   3. whatever option happens to be first in the dropdown.
  const sel = $("#chat-backend");
  const [active, health] = await Promise.all([
    loadActive(),
    jget("/healthz").catch(() => null),
  ]);
  const wanted =
    (active && active.backend) ||
    (health && health.cortex_backend && health.cortex_backend !== "noop"
      ? health.cortex_backend
      : null);
  if (wanted && [...sel.options].some((o) => o.value === wanted)) {
    sel.value = wanted;
  }
  await loadCatalog(sel.value);
  await initSessions();
}
