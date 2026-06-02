# FinRAG

Local-first RAG chatbot for financial PDFs (quarterly reports, annual filings, balance sheets). Built as an internship project at FinSeal. Hybrid retrieval (BM25 + cosine) over a Chroma index, with table-aware chunking, OCR fallback, vision-model figure transcription, period/company-aware filtering, and a mode registry (Extract / Analyze / Compare) on top. Django + DRF backend with DB-backed multi-chat history; plain HTML/CSS/JS frontend.

---

## Stack

| Layer            | Choice                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------- |
| LLM              | `gpt-oss:20b-cloud` via Ollama Cloud (any local Ollama chat model works)                 |
| Embeddings       | `nomic-embed-text` (768-dim) with `search_query:` / `search_document:` task prefixes     |
| Vector store     | ChromaDB (cosine), persisted to disk                                                     |
| Keyword search   | `rank-bm25` via LangChain `BM25Retriever`                                                |
| Fusion           | LangChain `EnsembleRetriever` (RRF)                                                      |
| PDF parsing      | PyMuPDF (text) · pdfplumber + Camelot stream + img2table (tables) · Tesseract (OCR)      |
| Vision           | `granite3.2-vision:2b` via Ollama — figure/diagram transcription                         |
| Memory           | Query rewriter + last `HISTORY_TURNS` messages via `MessagesPlaceholder`                 |
| Backend          | Django + Django REST Framework, SQLite (`Chat` / `Message` tables)                       |
| Frontend         | Plain HTML/CSS/JS, `marked.min.js` vendored, Bloomberg/fintech theme                     |

Embeddings, vectorstore, OCR, tables, and figures run locally. Only the chat LLM is cloud (and is swappable for a local model).

---

## Repository layout

```
finrag/
├── data/<company>/*.pdf      # Source PDFs, grouped by company folder
├── vectorstore/              # Persisted Chroma collection (gitignored)
├── finrag_backend/           # Django project (settings, urls)
├── chat/                     # DRF app: models, serializers, views, urls, rag bridge
├── frontend/                 # index.html · style.css · app.js · marked.min.js
├── evals/qa.yaml + run.py    # YAML-driven regression suite
├── config.py                 # Paths, model names, chunking, retrieval params
├── embeddings.py             # Nomic task-prefix wrapper + cosine-Chroma factory
├── parsers.py                # Per-company filename → period metadata
├── ingest.py                 # PDF → (text|OCR) + tables → chunks → embeddings → Chroma
├── ingest_figures.py         # Restartable vision pass: type="figure" chunks
├── query.py                  # Hybrid retrieval + intent detection + RAG chain
├── modes.py                  # Mode registry: Extract / Analyze / Compare
└── manage.py
```

---

## Configuration (`config.py`)

| Setting                    | Default                  |
| -------------------------- | ------------------------ |
| `LLM_MODEL`                | `gpt-oss:20b-cloud`      |
| `EMBEDDING_MODEL`          | `nomic-embed-text`       |
| `CHUNK_SIZE` / `OVERLAP`   | `1000` / `200`           |
| `TOP_K`                    | `8` (modes can bump to 12) |
| `MMR_FETCH_K` / `LAMBDA`   | `24` / `0.7`             |
| `MAX_CONTEXT_CHUNKS`       | `24`                     |
| `HYBRID_BM25_WEIGHT`       | `0.5`                    |
| `HISTORY_TURNS`            | `6`                      |
| `OCR_*`                    | DPI 300, min 30 chars    |
| `TABLE_*`                  | plumber + Camelot stream + img2table; min 2×2 |
| `VISION_*`                 | granite3.2-vision:2b · DPI 144 · 120 s timeout |

Schema-affecting changes (embedding model, chunk size, ingest headers) require wiping `vectorstore/` and re-ingesting.

---

## Ingestion (`ingest.py`)

```
data/<company>/*.pdf
   ├── parsers.parse_filename(company, name)   # q1-2026.pdf → {company:"infosys", period:"Q1FY26", fy:26, q:1}
   ├── For each page:
   │     text = page.get_text() or _ocr_page(page) if too short
   │     emit Document(type="text", header+text, metadata={source, company, period, fy, q, page, ocr})
   ├── For each page, tables:
   │     OCR'd page  → img2table on pixmap
   │     native page → pdfplumber, fallback Camelot stream
   │     emit Document(type="table", header+markdown, metadata={..., table_engine})
   ├── Figures (separate restartable pass, ingest_figures.py):
   │     vision model on figure-bearing native pages → Document(type="figure", description, metadata={..., vision_model})
   ├── RecursiveCharacterTextSplitter(1000/200) on text only (tables kept whole)
   └── Chroma (cosine, nomic prefixes), batched embedding (BATCH=64) with per-batch try/except
```

Every chunk header prepended to `page_content` so it gets embedded too:
```
Document: <filename>
Company: <slug>          # e.g. infosys, riil
Period:   <label>        # e.g. Q3FY24, FY25
Page:     <N>            # tables only
```

Per-company filename parsers live in `parsers.py` (`PARSERS` registry): Infosys quarterlies (`q1-2026.pdf` → `Q1FY26`), RIIL annuals (`Annual-Report-2024-25.pdf` → `FY25`). New companies = add one parser + register.

Run:
```bash
rm -rf vectorstore && python ingest.py
python ingest_figures.py         # optional, slow, restartable
```

---

## Query pipeline (`query.py`)

```
question (+ history)
   │
   ├── rewrite_query()                          # follow-ups → standalone
   │
   ├── intent detection (on question + rewritten):
   │     detect_company_filter()                # token + phrase aliases ({"riil","reliance","rel industrial"} → "riil")
   │     detect_periods()                       # Q3FY24, Q3 FY 2024, third quarter of FY24, FY25, ...
   │     expand_period_range()                  # "from X to Y" / "X vs Y" / "X through Y" → closed interval (cap 16)
   │     _period_label_to_filter_kwargs()       # "Q3FY24"→period_filter ; "FY24"→fy_filter=24 (matches quarterly chunks)
   │     is_numeric / is_figure / is_board / detect_statement_targets()
   │
   ├── retrieve(question, company_filter, period_filter|fy_filter, mode.top_k):
   │     SINGLE-period  → one hybrid pass with hard filter
   │     MULTI-period   → fan-out: one hybrid pass per period
   │                       + per-period ANCHOR PROBE (see below)
   │     MULTI-company  → falls back to hybrid (no filter)
   │
   │     priority: anchor_docs → statement_bonus → board_bonus → type_bonus → per-source/period bonus → base
   │     deduped, INR-preferred sort, capped at MAX_CONTEXT_CHUNKS
   │
   ├── format_context()
   │     "[i] (filename p.N · company · PERIOD) [TYPE]\n<chunk>"
   │     _normalize_indian_numbers()   # collapse "9 83" → 983, comma-format
   │
   └── PROMPT[mode] | ChatOllama | StrOutputParser
```

### Currency-aware retrieval

Infosys press releases publish the same statement of operations TWICE — once in `(In ₹ crore...)` and once in `(in US $ millions)`. Same business, different scale (e.g. Q3FY24 revenue = ₹38,821 cr vs US$4,663 M). Pre-fix, the model conflated the two.

`_doc_currency(doc)` classifies each chunk as `inr` / `usd` / `None` via three fingerprints, in order:
1. Header marker (`(In ₹ crore...)` / `(in US $ millions)`)
2. EPS unit regex (`EPS (₹)` vs `EPS ($)`) — survives sub-chunks where the caption was split off
3. `$N` value pattern fallback

`_prefer_inr(docs)` stable-sorts INR first, unknown middle, USD last — never deletes.

### Anchor probe (multi-period fan-out)

The `MAX_CONTEXT_CHUNKS=24` cap can starve a period of its statement-of-operations chunk when MMR diversifies the candidate pool. Per period, an extra hybrid pass pulls 6 `type="table"` candidates against a fixed probe (`"Revenues Cost of Sales Gross profit Operating profit Net profit Basic EPS in rupees crore"`), filters to INR, picks the first chunk containing both `Revenues` and `Operating profit`, and prepends it. One guaranteed INR statement-of-ops chunk per requested period.

### Modes (`modes.py`)

Mode = one dict (`system_prompt`, `top_k`, `max_context_chunks`, `temperature`). Adding a mode = adding one entry; no other code changes.

| Mode      | top_k | Output shape                                                                 |
| --------- | ----- | ---------------------------------------------------------------------------- |
| `extract` | 8     | Verbatim figures with `[filename p.N]` citations. Terse.                     |
| `analyze` | 12    | `## Headline / ## Key observations / ## Risks & flags / ## Bottom line`      |
| `compare` | 12    | `## Framing / ## Comparison table / ## Deltas & interpretation / ## Bottom line` |

The base prompt enforces: Indian FY mapping (Q1=Apr–Jun, … Q3=Oct–Dec), currency discipline (one unit per answer, never mix ₹ with $), Indian-digit normalization, trust labeled totals, refuse rather than fabricate.

Mode is **per-message** and **user-picked** (no auto-routing). The frontend defaults the composer to the last mode used in the chat.

### Other intent paths (carried over from Phase 2)

- **Statement-targeted retrieval** — standalone/consolidated × P&L/BS/CF triggers whole-page (text + table) bonuses.
- **Board / people-listing** — fixed probe surfaces company-info pages.
- **Multi-author** — per-source slice when ≥2 known authors are named.
- **Numeric intent** — half the context window reserved for table chunks.
- **Figure intent** — figure chunks get an "AUTHORITATIVE TRANSCRIPTION" framing.

---

## Backend (`finrag_backend/` + `chat/`)

DRF API under `/api`:

| Method | Path                       | Body                       | Returns                                   |
| ------ | -------------------------- | -------------------------- | ----------------------------------------- |
| GET    | `/modes`                   | —                          | `{modes:[{id,label,description}], default}` |
| GET    | `/chats`                   | —                          | List of chats (id, title, message_count)  |
| POST   | `/chats`                   | `{title?}`                 | New chat                                  |
| GET    | `/chats/{id}`              | —                          | Chat + full message list                  |
| DELETE | `/chats/{id}`              | —                          | 204                                       |
| POST   | `/chats/{id}/messages`     | `{question, mode?}`        | `{user_message, assistant_message, rewritten_query}` |

Models: `Chat(id, title, created_at, updated_at)` + `Message(chat FK, role, content, sources JSON, flags JSON, mode, created_at)`. Title auto-derived from the first question. `chat/rag.py` is the thin bridge into `query.ask()`.

`/` serves `frontend/index.html`; the rest is static via `STATICFILES_DIRS = [BASE_DIR / "frontend"]`. CORS open in dev.

---

## Frontend (`frontend/`)

Plain HTML/CSS/JS — no framework, no build step, just `fetch()`. Bloomberg/fintech aesthetic: IBM Plex Mono + Inter, dark slate `#0a0e14`, amber + green data accents, tabular numerals. Markdown rendered via vendored `marked.min.js` (GFM tables + line breaks).

- Sidebar: chat list + `+ New Query`.
- Messages pane: per-role bubbles, mode badge on assistant rows, expandable Sources panel (📄 TEXT / 📊 TABLE / 🖼️ FIGURE).
- Composer: `<select id="mode-select">` (EXTRACT / ANALYZE / COMPARE) + input + send. Defaults to the last mode used in the chat.
- Flags caption (when present): `Rewrote → …`, `Company → infosys`, `Period → Q3FY24`, `Period range → Q3FY24,Q3FY25,Q3FY26`, `Multi-company → …`.

---

## Prerequisites

```bash
brew install tesseract ghostscript   # macOS; or apt-get on Linux
ollama pull nomic-embed-text
ollama pull granite3.2-vision:2b     # optional, figures only
# Chat LLM — pick ONE:
ollama signin                         # gpt-oss:20b-cloud (default, light on RAM)
# OR
ollama pull llama3.1:8b               # set LLM_MODEL in config.py for fully offline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

---

## End-to-end usage

```bash
# 1. Drop PDFs into data/<company>/
# 2. Build index
rm -rf vectorstore && python ingest.py
python ingest_figures.py             # optional

# 3. Run backend (serves API + frontend)
python manage.py runserver           # http://localhost:8000

# 4. Regression check
python evals/run.py --retrieval      # ~1s, no LLM
python evals/run.py                  # full, with LLM
```

CLI fallback: `python query.py "your question"`.

---

## Design decisions

- **Hybrid BM25 + vector with nomic task prefixes** — biggest single quality lever. BM25 nails entity / line-item queries that pure cosine misses.
- **Tables are first-class chunks**, Markdown-serialised, never split. Reserved seats in the context window for numeric questions.
- **Period metadata at ingest time.** Filename → period via per-company parser registry (`parsers.py`); the period is hard-filtered at retrieval, not left to the LLM to filter from text.
- **Annual labels translate to numeric `fy_filter`** so `FY24` matches chunks tagged `Q1FY24..Q3FY24` even when no annual-period chunk exists.
- **Mode registry over agentic routing.** User picks Extract / Analyze / Compare per message; no LLM router in the loop. Easier to reason about, cheaper, deterministic.
- **Currency-aware retrieval + INR-first sort.** Prompt rules alone weren't enough — Infosys publishes ₹-crore and US$-million versions of the same table on adjacent pages. Detect at the chunk level, demote USD.
- **Per-period anchor probe.** Guarantees the INR statement-of-operations chunk for every requested period survives the `MAX_CONTEXT_CHUNKS` cap in multi-period fan-out.
- **Vision pass is separate, restartable, idempotent.** Vision calls are slow and the local runner can wedge; persisting per-PDF + skipping done pages means a crash costs minutes, not the whole index.
- **DB-backed history.** `Chat` / `Message` rows replace Streamlit's per-tab `session_state`. Per-message `mode` stored so the frontend can default the composer correctly on re-open.

---

## Known limitations

- **No cross-encoder re-ranker.** RRF ensemble only; a re-ranker (`bge-reranker-base`) could push the right chunk to rank 1 on competitive numeric queries.
- **Single-collection index.** Fine at hundreds of PDFs; per-domain collections would help past that.
- **BM25 corpus in RAM.** Fine at this scale; needs rework at tens of thousands of chunks.
- **Camelot stream is noisier than pdfplumber.** Filtered by `TABLE_MIN_ROWS/COLS` + `table_engine` metadata for later down-weighting.
- **No Q4 data in the current Infosys corpus** — FY-level totals aren't extractable from Q1–Q3 alone.
- **Cloud LLM is not fully offline.** Default `gpt-oss:20b-cloud` ships prompts (with retrieved context) to Ollama Cloud. Switch `LLM_MODEL` to a local model for strict on-prem.
- **On-the-fly user upload** is deliberately deferred — the ingest pipeline assumes the `data/<company>/` layout.
