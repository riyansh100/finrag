# FinRAG

Local-first RAG chatbot for financial PDFs (quarterly reports, annual filings). Hybrid retrieval (BM25 + cosine) over a Chroma index, **cross-encoder reranker** as stage 2, table-aware chunking, OCR + vision fallbacks, period/company-aware filtering, and an Extract/Analyze/Compare mode registry. **On-the-fly PDF uploads** per chat — drop a file, it's indexed into its own collection and merged into retrieval alongside the curated corpus. Django + DRF backend, vanilla JS frontend.

On top of RAG sits a **three-layer analytics stack** (Redis cache → SQLite facts → RAG) that turns every answer into structured rows the bot reuses on future questions.

On top of *that* sits a **financial dashboard** — Plotly charts (revenue/margin/balance-sheet trends) rendered straight from the `MetricFact` table (SQL only, no LLM), plus prompt-driven charts in chat (`plot infosys revenue trend`).

---

## What's new (latest)

- **Atomized retrieval.** Multi-company / multi-period questions are decomposed into `(company × period)` atoms (`nlu.build_atoms`); each atom gets its own hard-filtered retrieval pass so no cell crowds out the others. Fixes the "one period/company dominated the context" failure.
- **Swappable LLM provider.** `config.LLM_PROVIDER` (`"ollama" | "openrouter"`) routes all 6 chat-model sites through one factory (`llm_provider.make_chat`). OpenRouter is OpenAI-compatible; JSON mode is translated per provider. Old Ollama lines kept commented for instant revert.
- **Fact backfill.** `backfill.py` populates `MetricFact` from existing corpus chunks (not query-time) — reuses the `facts.py` extractor, batches large docs, **caches per-doc to disk so it's resumable and quota is spent once**.
- **Dashboard + prompt charts.** `dashboard.py` (data layer) + `GET /api/dashboard` + Plotly frontend. Pure SQL/ORM, zero LLM. Metric detection: LLM constrained to canonical keys, with an alias lookup as fallback.
- **Self-verify (deterministic, 0 LLM).** `verify.py` traces every comma-grouped figure in an answer back to the retrieved context or `MetricFact`; the frontend shows a "✓ N figures traced" / "⚠ unverified" badge.
- **Eval suite rebuilt** for the infosys/riil corpus (17 cases — scores atom routing AND verify status): `evals/run.py --retrieval` fast path, full run 117/118.

---

## Architecture at a glance

```
SETUP (one-time)
  Source PDFs ──ingest+embed──▶ ChromaDB        (vector chunks)
  Source PDFs ──backfill (LLM)─▶ MetricFact      (SQL facts table)

LIVE (every question)
  question
     │
     ▼
  NLU (LLM)  who · when · what          ── companies/periods/metrics, intent
     │
     ├── chart request?  ──yes──▶ MetricFact (SQL) ──▶ caption + Plotly   [NO LLM]
     │
     └── no ──▶ Retrieve+rerank (ChromaDB) ──▶ LLM answer ──▶ self-verify  [LLM]
                                                    │              │
                                                    └─ saves facts ─┴─▶ MetricFact

  blue path = uses LLM · green path (charts, verify, SQL) = no LLM
```

---

## Stack

| Layer            | Choice                                                                              |
| ---------------- | ----------------------------------------------------------------------------------- |
| LLM              | Swappable via `LLM_PROVIDER`: OpenRouter (OpenAI-compatible) or Ollama (`minimax-m3:cloud`) |
| Embeddings       | `nomic-embed-text` (768-dim) with `search_query:` / `search_document:` prefixes     |
| Vector store     | ChromaDB (cosine), persisted to disk                                                |
| Keyword search   | `rank-bm25` via LangChain `BM25Retriever`                                           |
| Fusion           | LangChain `EnsembleRetriever` (RRF)                                                 |
| Reranker         | `BAAI/bge-reranker-base` cross-encoder via `sentence-transformers` (stage 2)        |
| PDF parsing      | PyMuPDF (text) · pdfplumber + Camelot stream + img2table (tables) · Tesseract (OCR) |
| Vision           | `granite3.2-vision:2b` via Ollama — figure/diagram transcription                    |
| Memory (chat)    | LLM rewriter + last `HISTORY_TURNS` messages                                        |
| Memory (facts)   | Redis (L1) → SQLite `MetricFact` (L2) → RAG (L3)                                    |
| Backend          | Django + DRF, SQLite (`Chat`, `Message`, `MetricFact`, `AnalysisNote`)              |
| Dashboard        | `dashboard.py` (SQL over `MetricFact`) + Plotly.js (CDN) — charts, no LLM           |
| Frontend         | Plain HTML/CSS/JS, `marked.min.js` vendored                                         |

Everything except the chat LLM runs locally. Redis is optional — the cache silently falls back to SQLite if it's unreachable.

---

## Repository layout

```
finrag/
├── data/<company>/*.pdf      # Source PDFs, grouped by company folder
├── uploads/<upload_id>/      # On-the-fly uploaded PDFs (gitignored, per-chat)
├── vectorstore/              # Persisted Chroma collection (gitignored)
├── finrag_backend/           # Django project (settings, urls)
├── chat/                     # DRF app: models, views, urls, rag bridge
├── frontend/                 # index.html · style.css · app.js
├── evals/qa.yaml + run.py    # YAML regression suite
├── config.py                 # Paths, model names, retrieval + cache params
├── embeddings.py             # Nomic task-prefix wrapper + Chroma factory
├── parsers.py                # Per-company filename → period metadata
├── ingest.py                 # PDF → chunks → embeddings → Chroma
├── ingest_figures.py         # Restartable vision pass for figures
├── uploads.py                # Per-upload ingest, Chroma collection, retrieval merge
├── reranker.py               # Cross-encoder reranker (stage 2 of retrieval)
├── query.py                  # Hybrid retrieval + intent + RAG chain + cache hook
├── modes.py                  # Mode registry: Extract / Analyze / Compare
├── nlu.py                    # LLM slot extraction + build_atoms() (with regex fallback)
├── facts.py                  # Post-answer fact extractor + persistence
├── backfill.py               # Ingest-time MetricFact backfill from corpus (cached, resumable)
├── dashboard.py              # SQL → chart-ready series + prompt-driven chart specs (no LLM)
├── verify.py                 # Deterministic figure-tracing (answer → sources / MetricFact)
├── llm_provider.py           # Provider factory: OpenRouter | Ollama (config.LLM_PROVIDER)
├── cache.py                  # Redis L1 + SQLite L2 fact cache
├── recall.py                 # Scope-overlap match for "related past analysis"
└── manage.py
```

---

## How a question flows

```
question
  ├─ rewrite_query()                     # follow-ups → standalone; skipped when self-contained
  │
  ├─ nlu.extract_slots()                 # LLM JSON: {companies, quarters, fys, metrics, statement}
  │                                      # validated against corpus whitelist; regex fallback on failure
  │
  ├─ cache.lookup_for_slots()            # for each (company, period, metric) needed:
  │     ① Redis SCAN  → hit? use it
  │     ② SQLite      → hit? use it, warm Redis
  │     ③ neither     → fact uncached
  │
  ├─ decide:
  │     FULL coverage      → skip RAG, hand LLM the cached chunk only
  │     PARTIAL / no cover → run RAG, pin cached chunk to top of context
  │
  ├─ retrieve()                          # hybrid BM25 + vector, per-period fan-out,
  │                                      # statement-target anchor probes, INR-first sort,
  │                                      # USD duplicates dropped when INR exists
  │
  ├─ retrieve_from_uploads()             # if any PDFs are attached to this chat:
  │                                      # similarity search per upload collection,
  │                                      # statement-anchor probes, merged with corpus docs
  │                                      # (reserved quota so uploads aren't crowded out)
  │
  ├─ reranker.rerank()                   # stage 2: cross-encoder rescores the candidate
  │                                      # shortlist; anchors/statement-bonus chunks are
  │                                      # pinned at the front by content key
  │
  ├─ PROMPT[mode] | llm_provider.make_chat()  # mode = extract | analyze | compare
  │                                      # provider = openrouter | ollama (config.LLM_PROVIDER)
  │
  ├─ verify.verify_answer()              # deterministic: trace every comma-grouped figure in the
  │                                      # answer to the context / MetricFact → "traced" or "UNVERIFIED"
  │                                      # (no LLM; renders a badge under the answer)
  │
  ├─ facts.process_assistant_message()   # 2nd LLM pass extracts {company, period, metric, value, unit}
  │                                      # → MetricFact upsert + FactProvenance log + Redis write-through
  │                                      # → AnalysisNote(scope, body) for recall
  │
  └─ recall.find_candidates()            # weighted-Jaccard over past AnalysisNote scopes
                                          # surfaces top matches in the frontend recall panel
```

---

## The three cache layers

A fact = one row `(company, period, metric, value, unit, source_doc, page)`.

| Tier      | Lookup speed | When it has the answer                      | Fallback           |
| --------- | ------------ | ------------------------------------------- | ------------------ |
| **Redis** | ~ms          | Seen this exact fact recently (TTL 24h)     | → SQLite           |
| **SQLite** (`MetricFact`) | ~10ms        | Ever extracted this fact before; warms Redis on hit | → RAG              |
| **RAG**   | ~seconds     | Never seen it — go read the PDF             | LLM answers "n/a" if even RAG misses |

**Coverage gate** decides whether to skip RAG:
- needed = `companies × periods × metrics`
- if cache covers all of them → RAG skipped, LLM only sees the cached chunk (instructed to use values verbatim)
- partial → RAG runs for the missing ones, cached values pinned at top
- none → normal RAG

This is what makes the bot compounding: every answer writes new facts → next similar question hits the cache → faster + more deterministic.

---

## Recall (related past analyses)

Every assistant turn is mirrored into `AnalysisNote(scope, body_md)` where scope = `{companies, periods, statement}`. When a new question comes in, we score every past note:

```
score = 0.5 × period_overlap     (Jaccard)
      + 0.3 × company_overlap    (Jaccard, cross-company hard-zeros)
      + 0.2 × statement_match    (1.0 if same, 0.5 if either unspecified)
```

Notes scoring ≥ 0.5 surface as a cyan "Related past analysis" panel above the new answer, with **Show full answer** and **Re-ask** actions.

Cache coverage and recall scoring are **separate things**:
- **Cache coverage** (a count) decides whether the **LLM** runs RAG.
- **Recall score** (a percentage) decides whether the **user** sees the panel.

---

## NLU / query understanding

Replaced a brittle regex stack with an LLM JSON call that returns typed slots:

```python
nlu.extract_slots("infy topline last 3 fiscals")
# → {companies: ["infosys"], quarters: [], fys: [24, 25, 26],
#    metrics: ["revenue"], statement_variant: null, intent: "trend"}
```

- Validates against corpus whitelist (known companies + FYs in `data/`).
- Resolves carry-over from chat history ("the same comparison" → inherits company).
- Falls back to the original regex detectors (`_QUARTER_PATTERNS`, `_COMPANY_TOKEN_ALIASES`, etc.) on any failure.
- The rewriter is now self-contained-aware: skipped when the question already names its entities, and forbidden from inventing new ones.

---

## Configuration (`config.py`)

| Setting                          | Default                          |
| -------------------------------- | -------------------------------- |
| `LLM_PROVIDER`                   | `openrouter` (or `ollama`)       |
| `OPENROUTER_MODEL` / `OPENROUTER_API_KEY` | model slug / from `$OPENROUTER_API_KEY` env |
| `LLM_MODEL`                      | `minimax-m3:cloud` (used when provider=ollama) |
| `LLM_REQUEST_TIMEOUT_SEC`        | `120` (hard ceiling on every LLM call) |
| `EMBEDDING_MODEL`                | `nomic-embed-text`               |
| `CHUNK_SIZE` / `OVERLAP`         | `1000` / `200`                   |
| `TOP_K`                          | `8` (modes can bump to 12)       |
| `MAX_CONTEXT_CHUNKS`             | `24`                             |
| `HYBRID_BM25_WEIGHT`             | `0.5`                            |
| `RERANKER_ENABLED` / `RERANKER_MODEL` | `True` / `BAAI/bge-reranker-base` |
| `RERANKER_FETCH_K`               | `50` (candidates fed to the cross-encoder) |
| `HISTORY_TURNS`                  | `6`                              |
| `REDIS_URL`                      | `redis://localhost:6379/0` (`""` to disable) |
| `FACT_CACHE_TTL_SEC`             | `86400` (24h)                    |
| `FACT_CACHE_SHORTCIRCUIT_RAG`    | `True`                           |
| `UPLOAD_DIR` / `UPLOAD_MAX_MB`   | `BASE_DIR/uploads/` / `25`       |
| `UPLOAD_TOP_K`                   | `12` (chunks pulled per upload per question) |
| `UPLOAD_CONTEXT_FRACTION`        | `0.66` (reserved share of MAX_CONTEXT_CHUNKS for upload chunks) |
| `UPLOAD_FIGURE_DESCRIPTIONS`     | `False` (vision off for uploads — text + tables only) |

Schema-affecting changes (embedding model, chunk size, ingest headers) require wiping `vectorstore/` and re-ingesting.

---

## API (`/api`)

| Method | Path                       | Body                       | Returns                                         |
| ------ | -------------------------- | -------------------------- | ----------------------------------------------- |
| GET    | `/modes`                   | —                          | `{modes, default}`                              |
| GET    | `/dashboard?company=<slug>` | —                         | Chart-ready series from `MetricFact` (SQL, no RAG) |
| GET    | `/chats`                   | —                          | List of chats                                   |
| POST   | `/chats`                   | `{title?}`                 | New chat                                        |
| GET    | `/chats/{id}`              | —                          | Chat + full message list                        |
| DELETE | `/chats/{id}`              | —                          | 204                                             |
| POST   | `/chats/{id}/messages`     | `{question, mode?, upload_ids?}` | `{user_message, assistant_message, recall, rewritten_query, chart, verification}` |
| GET    | `/chats/{id}/uploads`      | —                          | List of `UploadedDoc` rows for this chat        |
| POST   | `/chats/{id}/uploads`      | `multipart: file=<pdf>`    | `UploadedDoc` row; indexes the PDF into its own Chroma collection |
| DELETE | `/chats/{id}/uploads/{upload_id}` | —                  | 204; drops collection, stored PDF, and the row  |
| GET    | `/recall?question=...`     | —                          | `{recall, scope}` — pre-submit lookup           |
| GET    | `/notes/{id}`              | —                          | Full `AnalysisNote` body                        |

Models: `Chat`, `Message`, `MetricFact`, `FactProvenance`, `AnalysisNote`, `UploadedDoc`.

---

## Modes (`modes.py`)

| Mode      | top_k | Output shape                                                                 |
| --------- | ----- | ---------------------------------------------------------------------------- |
| `extract` | 8     | Verbatim figures with `[filename p.N]` citations. Terse.                     |
| `analyze` | 12    | `## Headline / ## Key observations / ## Risks & flags / ## Bottom line`      |
| `compare` | 12    | `## Framing / ## Comparison table / ## Deltas & interpretation / ## Bottom line` |

Mode is per-message and user-picked. Base prompt enforces Indian FY mapping (Q1=Apr–Jun … Q3=Oct–Dec), currency discipline (never mix ₹ with $), refuse rather than fabricate, and treat `[CACHED-FACTS]` chunks as authoritative.

---

## Fact backfill (`backfill.py`)

Query-time extraction only fills `MetricFact` for what's been asked. Backfill front-fills it from the corpus so the dashboard has dense data:

```bash
python backfill.py --doc q1-2024.pdf      # dry run, one doc (prints facts)
python backfill.py --all --persist        # write every doc to MetricFact
```

- Reuses the validated `facts.py` extractor (feeds report chunk text + an explicit `company/period` hint).
- Batches large docs under a char budget (annual reports → several calls, tables only).
- **Caches each doc's facts to `backfill_cache/<doc>.json`** → re-runs and `--persist` cost no LLM; safe to Ctrl+C and resume (finished docs skip).

---

## Dashboard & charts (`dashboard.py`)

Pure SQL over `MetricFact` — **no RAG, no LLM, no cloud calls.**

- **Dashboard view** — sidebar 📊 button → `GET /api/dashboard` → Plotly charts (Revenue / Profitability / Margins % / Balance sheet) per company. Series are aligned to a chronological period axis, INR/USD collapsed to one line, gaps as nulls.
- **Prompt-driven charts** — `dashboard.chart_for_question()` detects chart intent (regex), a single company, and metric(s), and attaches a `chart` spec to the message response. Metric detection is **LLM-free** (regex over `facts.CANONICAL_METRICS` aliases), so charts pick the right metric even when the model is down.
- A metric needs ≥ 2 non-null points to auto-chart.

---

## Self-verify (`verify.py`)

After the LLM drafts an answer, every **comma-grouped figure** in it (e.g. `1,31,322`, `12,310`) is traced back to a source — **deterministically, with no extra LLM call**:

- Allowed set = every number in the retrieved context chunks **+** the `MetricFact` values for the question's company.
- A figure found in neither is flagged `UNVERIFIED`; the frontend renders a badge under the answer: `✓ 8/8 figures traced to sources` or `⚠ 1 figure unverified: …`.
- Format-tolerant (`1,31,322` ≡ `131,322`), and deliberately **conservative**: only absolute comma-grouped figures are hard-checked, so years, quarters, page numbers, and legitimately-computed percentages don't trip false alarms.
- Fault-tolerant and free — it only compares numbers already in hand, so it costs zero quota and never blocks the answer. Skipped for chart questions (the caption is already derived from `MetricFact`).

This is the cheap 80%-hallucination-killer: a wrong or invented figure surfaces immediately instead of slipping past in prose.

---

## On-the-fly PDF uploads

Drop a PDF onto the chat (or the paperclip button) and it's indexed into a **per-upload Chroma collection** named `upload_{id}`. The curated `data/` corpus is never touched.

- **Per-chat scope.** An upload belongs to one chat. Cross-chat access is blocked at the view layer.
- **Dedup.** SHA-256 of file bytes is unique per chat — re-attaching the same PDF returns the existing row instead of re-indexing.
- **Status FSM.** `pending → indexing → ready | failed`. UI polls and renders chips with a spinner / error state.
- **Filename → period stamp.** `detect_upload_meta()` parses `q1-2018.pdf` → `Q1FY18`, `Annual-Report-2024-25.pdf` → `FY25` so the LLM doesn't have to guess the period from raw page text.
- **Retrieval merge.** When a question is asked with upload IDs attached, the upload chunks are pulled by similarity (deterministic — no MMR) and statement-target anchor probes, then merged into the corpus context with a reserved quota (`UPLOAD_CONTEXT_FRACTION`) so they can't be crowded out.
- **Cleanup.** `pre_delete` signal on `UploadedDoc` (and on `Chat` cascade-delete) drops the Chroma collection and the stored PDF.

---

## Reranking (stage 2)

Hybrid BM25+vector is stage 1: cheap and shallow. Stage 2 feeds `[question, chunk]` pairs into a cross-encoder so the model attends to both jointly — far better at "in USD" / "Q3 FY24" / table-header semantics than bi-encoder cosine.

- Wired in at both `query.retrieve()` exits and `uploads.retrieve_from_uploads()`.
- Anchor and statement-bonus chunks are **pinned** at the front by a content key (`source`, `page`, `type`, content prefix) so deterministic guarantees (e.g. "the balance-sheet page must be in scope") are never overridden by raw question-similarity.
- Fail-open: if the model can't load or `.predict()` raises, retrieval returns the unsorted shortlist — the pipeline never breaks because of this stage.
- Toggle with `RERANKER_ENABLED=False` in `config.py` to bypass the model entirely.

---

## Prerequisites

```bash
brew install tesseract ghostscript redis    # macOS
brew services start redis                    # optional but recommended
ollama pull nomic-embed-text                 # embeddings (local, required)
ollama pull granite3.2-vision:2b             # optional, figures only

# Chat LLM — pick a provider in config.py (LLM_PROVIDER):
#   openrouter (default): export OPENROUTER_API_KEY="sk-or-v1-..."   # add to ~/.zshrc to persist
#   ollama:               ollama signin       # minimax-m3:cloud, or pull llama3.1:8b for offline

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

# 4. Inspect the cache
redis-cli keys 'fact:*'
```

CLI fallback: `python query.py "your question"`.

---

## Run with Docker

```bash
cp .env.example .env          # fill in OPENROUTER_API_KEY + DJANGO_SECRET_KEY
docker compose up --build     # starts web + redis
# open http://localhost:8000
```

The image is CPU-only (CPU PyTorch, no CUDA) and drops the legacy Streamlit UI.
Runtime state (`db.sqlite3`, `vectorstore/`, `uploads/`, `data/`) is mounted as
volumes, so it persists across rebuilds and is not baked into the image.

> **Ollama is still required, even with OpenRouter.** The chat LLM can be
> OpenRouter, but **embeddings always run through Ollama** (`nomic-embed-text`).
> The container reaches Ollama on the host via `host.docker.internal:11434`, so
> Ollama must be **running on your machine** (`ollama serve`) with the model
> pulled (`ollama pull nomic-embed-text`). If it isn't, every query 500s with
> `ConnectionError: Failed to connect to Ollama`.

After editing `.env`, recreate the container so it reloads the values —
a running container does **not** see `.env` edits:

```bash
docker compose up -d --force-recreate web
```

---

## Design decisions

- **Two-stage retrieval (hybrid → cross-encoder).** Stage 1 (BM25 + vector + RRF) pulls a wide candidate shortlist; stage 2 (`bge-reranker-base`) rescores `(question, chunk)` jointly and keeps the best. Single biggest accuracy lift after hybrid itself, no chunking changes.
- **Hybrid BM25 + vector + RRF.** BM25 nails entity / line-item queries pure cosine misses.
- **On-the-fly uploads as a first-class path.** Per-chat Chroma collections, filename-based period stamping, reserved quota in the context window so uploaded PDFs aren't crowded out by the corpus. Indexed once on attach, retrieval merge on every question.
- **Tables as first-class chunks** in Markdown, never split. Reserved seats in the context window for numeric questions.
- **Period metadata at ingest time** (filename → `Q3FY24`). Hard-filtered at retrieval, not left to the LLM.
- **LLM-based slot extraction** with regex fallback. Handles paraphrases ("infy topline"), relative dates ("last 3 fiscals"), and carry-over from history. Regex still wins on cost when the question is unambiguous.
- **Three-tier fact cache.** Redis → SQLite → RAG, with full-coverage short-circuit. Compounding: every answer fills the cache, so the bot gets faster the more it's used.
- **Authoritative cached values.** Cached facts are injected as a synthetic `[CACHED-FACTS]` chunk the LLM is instructed to use verbatim — eliminates re-reading errors on cells we've already validated.
- **Currency-aware retrieval.** Per-chunk INR/USD classifier (markers → EPS-symbol regex → magnitude fallback). USD chunks dropped when their INR twin exists, so the LLM can't misread `US$1,640M` as `₹1,640 cr`.
- **Statement-target-aware anchor probes.** Balance-sheet / cash-flow / P&L queries each use their own keyword probe, with text chunks allowed to win when pdfplumber misses the table boundary.
- **Recall is structured Jaccard, not embeddings.** Scope is a typed tuple; structured match is sharper than vector similarity for this use case.
- **Deterministic self-verify, not an LLM judge.** Figures are traced to sources by number-matching against the context + `MetricFact` — zero extra LLM calls. The model does the fuzzy generation; cheap deterministic code does the checkable verification. Same split as metric detection (LLM constrained to canonical keys, alias lookup as fallback).
- **Fault tolerance throughout.** Cache, extractor, recall, reranker, verify — every layer wrapped in try/except. Worst case is "slower RAG", never a broken answer.
- **Bounded LLM calls.** `LLM_REQUEST_TIMEOUT_SEC` is applied to every `ChatOllama` instance so a hung cloud endpoint surfaces a friendly "unreachable" message instead of an infinite spinner.

---

## Known limitations

- **Single corpus collection.** Curated `data/<company>/` PDFs share one Chroma collection; fine at hundreds of PDFs, per-domain collections beyond that. (Uploads are already isolated in their own per-upload collections.)
- **BM25 corpus in RAM.** Rework needed at tens of thousands of chunks.
- **No Q4 data in the current Infosys corpus** — FY totals not derivable from Q1–Q3 alone.
- **Cloud LLM is not strictly offline.** Default ships context to Ollama Cloud; switch `LLM_MODEL` for fully local.
- **Reranker first call downloads ~280 MB** of model weights into the HuggingFace cache on the box. Cached after.
- **Cache values are authoritative.** If a source PDF is updated, stale facts won't auto-refresh until TTL expires or you wipe (`redis-cli --scan --pattern 'fact:*' | xargs redis-cli del` + re-ask).
