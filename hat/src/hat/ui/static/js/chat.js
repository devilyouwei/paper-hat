import { $, $$, escapeHtml, renderBubbleHtml } from "./util.js";
import { jget, jpost, jdelete } from "./api.js";
import { loadCatalog, loadActive } from "./models.js";

let currentSessionId = null;

function chatbox() { return $("#chatbox"); }

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
  el.addEventListener("scroll", () => {
    stickToBottom = isNearBottom(el);
  }, { passive: true });
}

function appendBubble(role, content) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.innerHTML = renderBubbleHtml(content, { showThink: $("#show-thinking")?.checked });
  chatbox().appendChild(div);
  // A new bubble means a new turn — always pin to bottom and re-arm the lock.
  stickToBottom = true;
  scrollToBottom(true);
  return div;
}

function clearChat() { chatbox().innerHTML = ""; }

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
  for (const it of detail.messages) {
    if (it.query) appendBubble("user", it.query);
    if (it.response) appendBubble("assistant", it.response);
  }
  $$("#session-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.id === id),
  );
}

async function newSession() {
  try {
    const created = await jpost("/api/sessions", {});
    currentSessionId = created.id;
    clearChat();
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
  const correction = $("#correction").value.trim();
  $("#msg").value = "";
  $("#send-btn").disabled = true;

  appendBubble("user", text);
  const assistantDiv = appendBubble("assistant", "");
  let buffer = "";

  const body = {
    model: "hat-cortex",
    messages: [{ role: "user", content: text }],
    stream: true,
    temperature: parseFloat($("#temp").value),
    max_tokens: parseInt($("#max-tokens").value, 10),
    chat_template_kwargs: { enable_thinking: $("#enable-thinking").checked },
    session_id: currentSessionId,
  };
  if (correction) body.hat_correction = correction;

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
          const delta = obj.choices?.[0]?.delta?.content || "";
          if (delta) {
            buffer += delta;
            assistantDiv.innerHTML = renderBubbleHtml(buffer, { showThink });
            scrollToBottom();   // only scrolls if user is still near bottom
          }
        } catch {
          // ignore non-JSON keepalives
        }
      }
    }
    $("#correction").value = "";
    await loadSessions(currentSessionId);
  } catch (e) {
    assistantDiv.textContent = `network error: ${e.message}`;
  } finally {
    $("#send-btn").disabled = false;
  }
}

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
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });

  // Initialise the backend selector. Priority:
  //   1. backend hosting the currently-active model (post-refresh state).
  //   2. backend configured via HAT_CORTEX_BACKEND in the env (/healthz).
  //   3. whatever option happens to be first in the dropdown.
  const active = await loadActive();
  if (active && active.backend) {
    $("#chat-backend").value = active.backend;
  } else if (window.__hatEnvBackend) {
    const sel = $("#chat-backend");
    if ([...sel.options].some((o) => o.value === window.__hatEnvBackend)) {
      sel.value = window.__hatEnvBackend;
    }
  }
  await loadCatalog($("#chat-backend").value);
  await initSessions();
}
