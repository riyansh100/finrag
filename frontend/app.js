// FinRAG frontend — talks only to the Django API. No framework, just fetch().
//
// API contract (all under /api):
//   GET    /chats                  -> [ {id, title, message_count, ...}, ... ]
//   POST   /chats        {title?}  -> {id, title, ...}
//   GET    /chats/{id}             -> {id, title, messages: [ {role, content, sources, flags} ]}
//   DELETE /chats/{id}             -> 204
//   POST   /chats/{id}/messages {question}
//                                  -> {user_message, assistant_message, rewritten_query}

const API = "/api";

// --- tiny DOM helpers --------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
};

// Render assistant text as Markdown -> HTML. marked is loaded globally via
// marked.min.js (vendored, offline). GFM gives us tables + line breaks.
if (window.marked) {
  marked.setOptions({ gfm: true, breaks: true });
}
function renderMarkdown(text) {
  return window.marked ? marked.parse(text) : text;
}

// --- app state ---------------------------------------------------------------
let activeChatId = null; // which chat is open in the message pane

// --- API calls ---------------------------------------------------------------
async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok && res.status !== 204) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.status === 204 ? null : res.json();
}

const listChats = () => api("/chats");
const createChat = () => api("/chats", { method: "POST", body: "{}" });
const getChat = (id) => api(`/chats/${id}`);
const deleteChat = (id) => api(`/chats/${id}`, { method: "DELETE" });
const postMessage = (id, question) =>
  api(`/chats/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });

// --- sidebar rendering -------------------------------------------------------
async function refreshSidebar() {
  const chats = await listChats();
  const list = $("#chat-list");
  list.innerHTML = "";

  chats.forEach((chat) => {
    const li = el("li", "chat-item");
    if (chat.id === activeChatId) li.classList.add("active");

    const title = el("span", "title");
    title.textContent = chat.title || "New chat";
    li.appendChild(title);

    const del = el("button", "delete");
    del.textContent = "✕";
    del.title = "Delete chat";
    del.addEventListener("click", async (e) => {
      e.stopPropagation(); // don't also open the chat
      await deleteChat(chat.id);
      if (chat.id === activeChatId) {
        activeChatId = null;
        clearMessages();
        setComposerEnabled(false);
      }
      refreshSidebar();
    });
    li.appendChild(del);

    li.addEventListener("click", () => openChat(chat.id));
    list.appendChild(li);
  });
}

// --- message pane rendering --------------------------------------------------
function clearMessages() {
  $("#messages").innerHTML =
    '<div class="empty-state">Select a chat or start a new one to begin.</div>';
}

function renderSources(sources) {
  const details = el("details", "sources");
  const summary = el("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);

  sources.forEach((s, i) => {
    const wrap = el("div", "source");
    const tag =
      { table: "📊 TABLE", figure: "🖼️ FIGURE" }[s.type] || "📄 TEXT";
    const head = el("div", "head");
    head.textContent = `[${i + 1}] ${s.source} — p.${s.page} · ${tag}`;
    const pre = el("pre");
    pre.textContent = s.content;
    wrap.appendChild(head);
    wrap.appendChild(pre);
    details.appendChild(wrap);
  });
  return details;
}

function addMessage({ role, content, sources = [], flags = [] }) {
  // remove the empty-state placeholder if present
  const empty = $("#empty-state") || document.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = el("div", `msg ${role}`);

  const roleLabel = el("div", "role");
  roleLabel.textContent = role;
  msg.appendChild(roleLabel);

  if (flags && flags.length) {
    const f = el("div", "flags");
    f.textContent = flags.join(" · ");
    msg.appendChild(f);
  }

  const bubble = el("div", "bubble");
  if (role === "assistant") {
    // Assistant answers are Markdown (bold, tables, lists) -> render to HTML.
    bubble.classList.add("markdown");
    bubble.innerHTML = renderMarkdown(content);
  } else {
    // User text is shown verbatim (textContent escapes any HTML).
    bubble.textContent = content;
  }
  msg.appendChild(bubble);

  if (sources && sources.length) {
    msg.appendChild(renderSources(sources));
  }

  $("#messages").appendChild(msg);
  scrollToBottom();
  return msg;
}

function scrollToBottom() {
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
}

// --- actions -----------------------------------------------------------------
async function openChat(id) {
  activeChatId = id;
  setComposerEnabled(true);
  await refreshSidebar(); // re-highlight active item

  const chat = await getChat(id);
  $("#messages").innerHTML = "";
  if (!chat.messages.length) {
    clearMessages();
  } else {
    chat.messages.forEach(addMessage);
  }
}

async function startNewChat() {
  const chat = await createChat();
  await refreshSidebar();
  await openChat(chat.id);
  $("#question-input").focus();
}

async function sendQuestion(question) {
  // If no chat is open, create one on the fly.
  if (!activeChatId) {
    const chat = await createChat();
    activeChatId = chat.id;
  }

  addMessage({ role: "user", content: question });

  // optimistic "thinking" bubble
  const pending = addMessage({ role: "assistant", content: "" });
  pending.classList.add("pending");

  setComposerEnabled(false);
  try {
    const res = await postMessage(activeChatId, question);
    pending.remove();
    addMessage(res.assistant_message);
    await refreshSidebar(); // title may have auto-updated; reorder by recency
  } catch (err) {
    pending.classList.remove("pending");
    pending.querySelector(".bubble").textContent = "⚠️ " + err.message;
  } finally {
    setComposerEnabled(true);
    $("#question-input").focus();
  }
}

function setComposerEnabled(enabled) {
  $("#question-input").disabled = !enabled;
  $("#send-btn").disabled = !enabled;
}

// --- wire up events ----------------------------------------------------------
$("#new-chat-btn").addEventListener("click", startNewChat);

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("#question-input");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  sendQuestion(q);
});

// --- init --------------------------------------------------------------------
(async function init() {
  setComposerEnabled(true); // allow asking even before picking a chat
  await refreshSidebar();
})();
