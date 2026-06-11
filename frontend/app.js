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
// Slice-4: uploads attached to the active chat. Re-fetched on openChat().
// Shape: [{id, filename, status, pages, chunk_count, error, ...}]. Pending
// uploads (mid-fetch) are spliced in optimistically with status="indexing"
// and a client-only `_localId` for re-render targeting.
let activeUploads = [];

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
const postMessage = (id, question, mode, uploadIds) =>
  api(`/chats/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({
      question, mode,
      upload_ids: uploadIds && uploadIds.length ? uploadIds : undefined,
    }),
  });
const getNote = (id) => api(`/notes/${id}`);
const getDashboard = (company) =>
  api(`/dashboard${company ? `?company=${encodeURIComponent(company)}` : ""}`);

// Slice-4 uploads. Multipart POST goes through fetch directly (api() forces
// JSON content-type which breaks multipart boundary detection).
const listUploads  = (chatId) => api(`/chats/${chatId}/uploads`);
const deleteUpload = (chatId, uploadId) =>
  api(`/chats/${chatId}/uploads/${uploadId}`, { method: "DELETE" });
async function uploadPdf(chatId, file) {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(`${API}/chats/${chatId}/uploads`, {
    method: "POST",
    body: form,
  });
  const body = await res.json().catch(() => ({}));
  // 200 = dedup hit (same file already attached), 201 = freshly indexed.
  if (!res.ok && res.status !== 200) {
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return body;
}

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
        activeUploads = [];
        renderAttachedStrip();
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

// --- Slice-4: uploads UI -----------------------------------------------------
// The composer has a paperclip + drag-drop on the message pane. Each upload
// becomes a chip in #attached-strip. Chips render in three states:
//   indexing  -> spinner, no remove (we're still POST'ing)
//   ready     -> coloured chip, ✕ removes from chat
//   failed    -> red chip showing the error message
// On send, every READY upload's id is sent in the message body so the backend
// merges that PDF's per-upload Chroma collection with the corpus retrieval.

function renderAttachedStrip() {
  const strip = $("#attached-strip");
  strip.innerHTML = "";
  if (!activeUploads.length) { strip.hidden = true; return; }
  strip.hidden = false;

  activeUploads.forEach((u) => {
    const chip = el("div", `upload-chip ${u.status || "ready"}`);

    if (u.status === "indexing" || u.status === "pending") {
      chip.appendChild(el("span", "spinner"));
    }
    const name = el("span", "name");
    name.textContent = u.filename;
    name.title = u.filename;
    chip.appendChild(name);

    if (u.status === "ready") {
      const meta = el("span", "meta");
      meta.textContent = `${u.pages}p · ${u.chunk_count}c`;
      chip.appendChild(meta);
    } else if (u.status === "indexing" || u.status === "pending") {
      const meta = el("span", "meta");
      meta.textContent = "indexing…";
      chip.appendChild(meta);
    } else if (u.status === "failed") {
      const meta = el("span", "meta");
      meta.textContent = (u.error || "failed").slice(0, 60);
      meta.title = u.error || "failed";
      chip.appendChild(meta);
    }

    // Remove button: only meaningful for rows that have a server-side id
    // (i.e. not the optimistic pending placeholder before POST returns).
    if (u.id !== undefined) {
      const rm = el("button", "remove");
      rm.type = "button";
      rm.textContent = "✕";
      rm.title = "Detach this PDF from the chat";
      rm.addEventListener("click", () => removeUpload(u.id));
      chip.appendChild(rm);
    }

    strip.appendChild(chip);
  });
}

async function refreshAttachedStrip() {
  if (!activeChatId) { activeUploads = []; renderAttachedStrip(); return; }
  try {
    activeUploads = await listUploads(activeChatId) || [];
  } catch (e) {
    console.error("listUploads failed:", e);
    activeUploads = [];
  }
  renderAttachedStrip();
}

async function handleFileSelected(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name)) {
    alert("Only PDF files can be attached.");
    return;
  }
  // Ensure we have a chat to attach into. If the user hasn't opened/created
  // one yet, mint one now so the upload has a home.
  if (!activeChatId) {
    const chat = await createChat();
    activeChatId = chat.id;
    await refreshSidebar();
  }

  // Optimistic placeholder chip while the server indexes the PDF.
  const localId = `local-${Date.now()}`;
  activeUploads.push({
    _localId: localId,
    filename: file.name,
    status: "indexing",
  });
  renderAttachedStrip();

  try {
    const row = await uploadPdf(activeChatId, file);
    // Replace the placeholder with the real row (or merge — backend may have
    // returned an existing dedup hit). Drop placeholder by _localId.
    activeUploads = activeUploads.filter((u) => u._localId !== localId);
    // If the same upload id is already in the list (dedup), don't duplicate.
    if (!activeUploads.some((u) => u.id === row.id)) {
      activeUploads.push(row);
    }
  } catch (err) {
    console.error("upload failed:", err);
    const placeholder = activeUploads.find((u) => u._localId === localId);
    if (placeholder) {
      placeholder.status = "failed";
      placeholder.error = err.message || "upload failed";
    }
  }
  renderAttachedStrip();
}

async function removeUpload(uploadId) {
  if (!activeChatId) return;
  // Optimistic remove — restore on error.
  const before = activeUploads;
  activeUploads = activeUploads.filter((u) => u.id !== uploadId);
  renderAttachedStrip();
  try {
    await deleteUpload(activeChatId, uploadId);
  } catch (e) {
    console.error("deleteUpload failed:", e);
    activeUploads = before;
    renderAttachedStrip();
  }
}

function readyUploadIds() {
  return activeUploads
    .filter((u) => u.status === "ready" && u.id !== undefined)
    .map((u) => u.id);
}

// --- dashboard (cross-document charts) ---------------------------------------
// Pure read of /api/dashboard (SQL over MetricFact, no RAG). Renders one
// Plotly chart per default grouping, per company. Series come pre-aligned to a
// shared period axis with nulls for gaps, so we plot them directly.

const CHART_COLORS = ["#2ecc71", "#38bdf8", "#f5a623", "#ff5c5c", "#a78bfa"];

const PLOT_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "'IBM Plex Mono', monospace", color: "#8a97a8", size: 11 },
  margin: { l: 56, r: 16, t: 8, b: 36 },
  xaxis: { gridcolor: "#232c3b", zerolinecolor: "#232c3b" },
  yaxis: { gridcolor: "#232c3b", zerolinecolor: "#232c3b" },
  legend: { orientation: "h", y: -0.2, font: { size: 10 } },
  showlegend: true,
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function buildTraces(group, companyData) {
  const periods = companyData.periods;
  const traces = [];
  let ci = 0;
  (group.metrics || []).forEach((metric) => {
    const s = companyData.series[metric];
    if (!s || s.values.every((v) => v == null)) return; // nothing to plot
    const color = CHART_COLORS[ci++ % CHART_COLORS.length];
    traces.push(
      group.kind === "bar"
        ? { type: "bar", x: periods, y: s.values, name: s.label,
            marker: { color } }
        : { type: "scatter", mode: "lines+markers", x: periods, y: s.values,
            name: s.label, line: { color, width: 2 },
            marker: { color, size: 6 }, connectgaps: false }
    );
  });
  return traces;
}

// Render a prompt-driven chart spec ({company, title, kind, periods, series})
// as a Plotly card appended to a message. Returns the card node, or null if
// Plotly is unavailable / the spec is empty.
function renderChartCard(chart) {
  if (!chart || !window.Plotly || !chart.series?.length) return null;
  const card = el("div", "dash-card chat-chart");
  card.appendChild(Object.assign(el("div", "dash-card-title"),
    { textContent: chart.title }));
  const plot = el("div", "dash-plot");
  card.appendChild(plot);

  const traces = chart.series.map((s, i) => {
    const color = CHART_COLORS[i % CHART_COLORS.length];
    return chart.kind === "bar"
      ? { type: "bar", x: chart.periods, y: s.values, name: s.label,
          marker: { color } }
      : { type: "scatter", mode: "lines+markers", x: chart.periods,
          y: s.values, name: s.label, line: { color, width: 2 },
          marker: { color, size: 6 }, connectgaps: false };
  });
  // Plot after the node is attached so Plotly sizes to the container.
  setTimeout(() => Plotly.newPlot(plot, traces,
    { ...PLOT_LAYOUT, barmode: "group" }, PLOT_CONFIG), 0);
  return card;
}

async function openDashboard() {
  activeChatId = null;
  setComposerEnabled(false);
  await refreshSidebar(); // clears the active highlight
  activeUploads = [];
  renderAttachedStrip();

  const pane = $("#messages");
  pane.innerHTML = "";
  const wrap = el("div", "dashboard");
  wrap.appendChild(Object.assign(el("h2", "dash-title"),
    { textContent: "Financial Dashboard" }));
  pane.appendChild(wrap);

  if (!window.Plotly) {
    wrap.appendChild(Object.assign(el("div", "dash-empty"), {
      textContent: "⚠️ Charts need Plotly, which failed to load (offline?). "
        + "The data endpoint still works at /api/dashboard.",
    }));
    return;
  }

  let payload;
  try {
    payload = await getDashboard();
  } catch (e) {
    wrap.appendChild(Object.assign(el("div", "dash-empty"),
      { textContent: "⚠️ Failed to load dashboard: " + e.message }));
    return;
  }

  if (!payload.companies.length) {
    wrap.appendChild(Object.assign(el("div", "dash-empty"), {
      textContent: "No financial facts yet. Ask some numeric questions, or run "
        + "the backfill, to populate the dashboard.",
    }));
    return;
  }

  payload.companies.forEach((company) => {
    const cData = payload.data[company];
    const section = el("div", "dash-company");
    section.appendChild(Object.assign(el("h3", "dash-company-name"),
      { textContent: company.toUpperCase() }));

    const grid = el("div", "dash-grid");
    payload.default_charts.forEach((group) => {
      const traces = buildTraces(group, cData);
      if (!traces.length) return; // skip charts with no data for this company
      const card = el("div", "dash-card");
      card.appendChild(Object.assign(el("div", "dash-card-title"),
        { textContent: group.title }));
      const plot = el("div", "dash-plot");
      card.appendChild(plot);
      grid.appendChild(card);
      // Plot after the node is in the DOM so Plotly can size to the container.
      Plotly.newPlot(plot, traces,
        { ...PLOT_LAYOUT, barmode: "group" }, PLOT_CONFIG);
    });

    if (!grid.children.length) {
      section.appendChild(Object.assign(el("div", "dash-empty"),
        { textContent: "No chartable metrics for this company yet." }));
    } else {
      section.appendChild(grid);
    }
    wrap.appendChild(section);
  });
}

// --- actions -----------------------------------------------------------------
async function openChat(id) {
  activeChatId = id;
  setComposerEnabled(true);
  await refreshSidebar();
  // Load the chat's attached PDFs in parallel with the message list -- they
  // both come from the same chat and the user expects to see them together.
  refreshAttachedStrip();

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
    refreshAttachedStrip();
  }
  const mode = $("#mode-select").value || defaultMode;
  // Send every READY upload attached to this chat. The backend re-validates
  // against UploadedDoc.status so anything mid-index is silently ignored.
  const uploadIds = readyUploadIds();

  addMessage({ role: "user", content: question });

  const pending = addMessage({ role: "assistant", content: "", mode });
  pending.classList.add("pending");

  setComposerEnabled(false);
  try {
    const res = await postMessage(activeChatId, question, mode, uploadIds);
    pending.remove();
    // Recall first (above the new answer) so the user sees "we have prior
    // work on this" before the new answer they're about to read.
    const recallPanel = renderRecallPanel(res.recall, question);
    if (recallPanel) $("#messages").appendChild(recallPanel);
    const msgNode = addMessage(res.assistant_message);
    // Prompt-driven chart: drop a Plotly card into the assistant bubble when
    // the backend resolved one for this question.
    const chartCard = renderChartCard(res.chart);
    if (chartCard) { msgNode.appendChild(chartCard); scrollToBottom(); }
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
  $("#attach-btn").disabled = !enabled;
}

// --- wire up events ----------------------------------------------------------
$("#new-chat-btn").addEventListener("click", startNewChat);
$("#dashboard-btn").addEventListener("click", openDashboard);

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("#question-input");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  sendQuestion(q);
});

// --- Slice-4: attach button + drag-drop -------------------------------------
$("#attach-btn").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  e.target.value = "";  // allow re-picking the same file
  if (file) handleFileSelected(file);
});

// Drag-and-drop on the message pane. We toggle a class on the overlay so it
// fades in only while a real drag is in progress. dragleave fires on every
// child boundary, so we track the depth with a counter.
let _dragDepth = 0;
const dropOverlay = $("#dropzone-overlay");
const mainPane = $("#main");

function _isFileDrag(e) {
  return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
}

mainPane.addEventListener("dragenter", (e) => {
  if (!_isFileDrag(e)) return;
  e.preventDefault();
  _dragDepth++;
  dropOverlay.classList.add("dragover");
});
mainPane.addEventListener("dragover", (e) => {
  if (!_isFileDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = "copy";
});
mainPane.addEventListener("dragleave", (e) => {
  if (!_isFileDrag(e)) return;
  _dragDepth--;
  if (_dragDepth <= 0) {
    _dragDepth = 0;
    dropOverlay.classList.remove("dragover");
  }
});
mainPane.addEventListener("drop", (e) => {
  if (!_isFileDrag(e)) return;
  e.preventDefault();
  _dragDepth = 0;
  dropOverlay.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
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
