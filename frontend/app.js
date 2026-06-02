// FinRAG frontend — talks only to the Django API. No framework, just fetch().
//
// API contract (all under /api):
//   GET    /modes                  -> {modes: [{id,label,description}], default}
//   GET    /chats                  -> [ {id, title, message_count, ...}, ... ]
//   POST   /chats        {title?}  -> {id, title, ...}
//   GET    /chats/{id}             -> {id, title, messages: [ {role, content, sources, flags, mode} ]}
//   DELETE /chats/{id}             -> 204
//   POST   /chats/{id}/messages {question, mode?}
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
let activeChatId = null;     // which chat is open in the message pane
let modes = [];              // [{id,label,description}] from /api/modes
let defaultMode = "extract"; // fallback if /api/modes hasn't loaded yet

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

const fetchModes = () => api("/modes");
const listChats = () => api("/chats");
const createChat = () => api("/chats", { method: "POST", body: "{}" });
const getChat = (id) => api(`/chats/${id}`);
const deleteChat = (id) => api(`/chats/${id}`, { method: "DELETE" });
const postMessage = (id, question, mode) =>
  api(`/chats/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ question, mode }),
  });
const getNote = (id) => api(`/notes/${id}`);

// --- mode dropdown -----------------------------------------------------------
function populateModeSelect() {
  const sel = $("#mode-select");
  sel.innerHTML = "";
  modes.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label.toUpperCase();
    opt.title = m.description;
    sel.appendChild(opt);
  });
  sel.value = defaultMode;
  sel.disabled = false;
}

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
      e.stopPropagation();
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
    '<div class="empty-state">◇ Select a query or start a new one to begin.</div>';
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

function modeBadge(modeId) {
  if (!modeId) return null;
  const m = modes.find((x) => x.id === modeId);
  const span = el("span", "mode-badge");
  span.dataset.mode = modeId;
  span.textContent = (m ? m.label : modeId).toUpperCase();
  return span;
}

function addMessage({ role, content, sources = [], flags = [], mode = "" }) {
  const empty = $("#empty-state") || document.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = el("div", `msg ${role}`);

  const roleRow = el("div", "role");
  const roleLabel = el("span");
  roleLabel.textContent = role;
  roleRow.appendChild(roleLabel);
  if (role === "assistant" && mode) {
    const badge = modeBadge(mode);
    if (badge) roleRow.appendChild(badge);
  }
  msg.appendChild(roleRow);

  if (flags && flags.length) {
    const f = el("div", "flags");
    f.textContent = flags.join(" · ");
    msg.appendChild(f);
  }

  const bubble = el("div", "bubble");
  if (role === "assistant") {
    bubble.classList.add("markdown");
    bubble.innerHTML = renderMarkdown(content);
  } else {
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

// --- recall panel (Slice 3) --------------------------------------------------
// Render above the assistant answer when the API returns related past
// analyses (overlapping company + period + statement scope). Lets the user
// see prior work and click "Re-ask" to refresh it against the current corpus.
function renderRecallPanel(recall, originalQuestion) {
  if (!recall || !recall.length) return null;
  const panel = el("div", "recall-panel");

  const head = el("div", "recall-head");
  head.textContent = `↺ Related past analysis (${recall.length})`;
  panel.appendChild(head);

  recall.forEach((r) => {
    const card = el("div", "recall-card");

    const meta = el("div", "recall-meta");
    const scopeBits = [];
    if (r.scope?.companies?.length) scopeBits.push(r.scope.companies.join(", "));
    if (r.scope?.periods?.length) scopeBits.push(r.scope.periods.join(", "));
    if (r.scope?.statement) scopeBits.push(r.scope.statement);
    const when = new Date(r.created_at).toLocaleString();
    meta.textContent = `${scopeBits.join(" · ")} · ${r.mode} · ${when} · match ${(r.score * 100).toFixed(0)}%`;
    card.appendChild(meta);

    const preview = el("div", "recall-preview markdown");
    preview.innerHTML = renderMarkdown(r.preview || "");
    card.appendChild(preview);

    // Lazy-expand to full body on demand (the API ships only 280 chars).
    if (r.body_length > (r.preview || "").length) {
      const moreBtn = el("button", "recall-action");
      moreBtn.textContent = "Show full answer";
      moreBtn.addEventListener("click", async () => {
        moreBtn.disabled = true;
        moreBtn.textContent = "Loading…";
        try {
          const full = await getNote(r.id);
          preview.innerHTML = renderMarkdown(full.body_md || "");
          moreBtn.remove();
        } catch (e) {
          moreBtn.textContent = "Error — retry";
          moreBtn.disabled = false;
        }
      });
      card.appendChild(moreBtn);
    }

    // Re-ask submits the ORIGINAL current question (not the past one) so the
    // user gets a fresh answer using whatever new facts are in the cache /
    // PDFs since the prior turn.
    const refreshBtn = el("button", "recall-action primary");
    refreshBtn.textContent = "Re-ask this question";
    refreshBtn.addEventListener("click", () => sendQuestion(originalQuestion));
    card.appendChild(refreshBtn);

    panel.appendChild(card);
  });
  return panel;
}

function scrollToBottom() {
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
}

// --- actions -----------------------------------------------------------------
async function openChat(id) {
  activeChatId = id;
  setComposerEnabled(true);
  await refreshSidebar();

  const chat = await getChat(id);
  $("#messages").innerHTML = "";
  if (!chat.messages.length) {
    clearMessages();
  } else {
    chat.messages.forEach(addMessage);
    // Default the composer to the LAST mode used in this chat (per-message
    // mode -- user can still switch for the next turn).
    const lastAssistant = [...chat.messages].reverse()
      .find((m) => m.role === "assistant" && m.mode);
    if (lastAssistant) $("#mode-select").value = lastAssistant.mode;
  }
}

async function startNewChat() {
  const chat = await createChat();
  await refreshSidebar();
  await openChat(chat.id);
  $("#question-input").focus();
}

async function sendQuestion(question) {
  if (!activeChatId) {
    const chat = await createChat();
    activeChatId = chat.id;
  }
  const mode = $("#mode-select").value || defaultMode;

  addMessage({ role: "user", content: question });

  const pending = addMessage({ role: "assistant", content: "", mode });
  pending.classList.add("pending");

  setComposerEnabled(false);
  try {
    const res = await postMessage(activeChatId, question, mode);
    pending.remove();
    // Recall first (above the new answer) so the user sees "we have prior
    // work on this" before the new answer they're about to read.
    const recallPanel = renderRecallPanel(res.recall, question);
    if (recallPanel) $("#messages").appendChild(recallPanel);
    addMessage(res.assistant_message);
    await refreshSidebar();
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
  $("#mode-select").disabled = !enabled || modes.length === 0;
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
  try {
    const data = await fetchModes();
    modes = data.modes || [];
    defaultMode = data.default || "extract";
    populateModeSelect();
  } catch (e) {
    console.error("Failed to load modes:", e);
  }
  setComposerEnabled(true);
  await refreshSidebar();
})();
