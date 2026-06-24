# FinRAG

**Ask plain-English questions about financial PDFs and get cited, fact-checked answers.**

FinRAG is a retrieval-augmented chatbot for quarterly results and annual reports. You point it at a folder of company PDFs (or drag one into the chat); it indexes them, and you can ask things like *"compare Infosys total assets across Q1 FY24 and Q1 FY25"* or *"give me the balance sheet for Q1 FY24."* Every figure in the answer is traced back to the source page, and questions like *"plot Infosys revenue trend"* render a chart straight from a structured facts table — no LLM guessing.

It's built to be **trustworthy on numbers**: a wrong figure in finance is worse than "not found," so the system refuses to fabricate, hard-filters by company/period, and verifies every number it prints.

```
You:  give me the balance sheet for infosys Q1 FY24
Bot:  ## Balance Sheet — Infosys, Q1 FY24 (₹ crore)
      Total assets        1,31,322   [q1-2024.pdf p.4]
      Total liabilities      56,494   [q1-2024.pdf p.4]
      ...
      ✓ 8/8 figures traced to sources
```

---

## What it can do

- **Hybrid retrieval + reranking** — keyword (BM25) and semantic (vector) search are fused, then a cross-encoder re-scores the shortlist for the final, sharper ranking.
- **Company/period aware** — "Q1 FY24" and "Infosys" become hard filters, so the model never mixes quarters or companies. Multi-company/multi-period questions are split into independent retrieval passes so no cell crowds out another.
- **Self-verifying answers** — every figure is deterministically traced back to a source chunk (no extra LLM call). Untraceable numbers get flagged.
- **Compounding fact cache** — answers are distilled into a `MetricFact` table; repeat questions are served from cache (Redis → SQLite) instead of re-reading the PDF.
- **Charts with no LLM** — a Plotly dashboard and in-chat charts are rendered purely from the facts table via SQL.
- **Drag-and-drop PDFs** — drop a file into a chat and it's indexed into its own isolated collection, merged into retrieval alongside the curated corpus.
- **Swappable LLM** — flip one setting between OpenRouter (cloud) and Ollama (local). Embeddings always run locally via Ollama.
- **Retargetable** — company/fiscal/metric config lives in a swappable [`packs/`](packs/) domain pack, so the same engine can point at a different corpus without code changes.

---

## Quick start (Docker)

The fastest way to run the whole thing — web app + Redis — in one command.

**Prerequisite: Ollama must be running on your machine**, even if you use OpenRouter for chat. FinRAG's *embeddings* always use Ollama's `nomic-embed-text` model:

```bash
ollama pull nomic-embed-text     # one-time, ~280 MB
ollama serve                     # leave running (or use the Ollama desktop app)
```

Then:

```bash
git clone <repo> && cd finrag
cp .env.example .env              # then edit it (see below)
docker compose up --build         # starts web + redis
```

Open **http://localhost:8000**.

In `.env` you must fill in two values:
- `OPENROUTER_API_KEY` — get one at https://openrouter.ai/keys (free models available).
- `DJANGO_SECRET_KEY` — generate with
  `python -c "from django.core.management.utils import get_random_secret_key as g; print(g())"`

> **Gotchas**
> - If queries return `500 ConnectionError: Failed to connect to Ollama`, Ollama isn't running/reachable. The container reaches it at `host.docker.internal:11434`.
> - After editing `.env`, a running container won't see the change — reload it:
>   `docker compose up -d --force-recreate web`
> - First query takes ~30–60s while the reranker model downloads (cached after, persisted in a volume).

To run it later (image already built): `docker compose up -d`. To stop: `docker compose down`.

---

## How it works

```
SETUP (one-time)
  Source PDFs ──ingest + embed──▶ ChromaDB     (searchable text/table chunks)
  Source PDFs ──backfill (LLM)──▶ MetricFact    (structured facts table)

LIVE (every question)
  question
     │
     ▼
  understand it   (LLM)         → which companies / periods / metrics, what intent
     │
     ├── chart request? ──yes──▶ MetricFact (SQL) ──▶ Plotly chart      [no LLM]
     │
     └── no ──▶ retrieve + rerank (ChromaDB) ──▶ LLM writes answer ──▶ verify figures
                                                      │                      │
                                                      └──── saves facts ─────┴──▶ MetricFact
```

A question flows through roughly these stages (all in [`query.py`](query.py)):

1. **Understand** — [`nlu.extract_slots`](nlu.py) asks the LLM for typed slots (`companies`, `quarters`, `fys`, `metrics`, `intent`), validated against what's actually in the corpus. Falls back to regex if the model misbehaves.
2. **Cache check** — if the needed `(company, period, metric)` facts are already in Redis/SQLite, skip retrieval and hand the LLM the cached values verbatim.
3. **Retrieve** — hybrid BM25 + vector search, with per-period/company fan-out and statement-aware probes (a balance-sheet question pulls the balance-sheet page).
4. **Rerank** — a cross-encoder ([`reranker.py`](reranker.py)) re-scores the shortlist; "must-include" pages are pinned so they can't be dropped.
5. **Answer** — the LLM writes the response under a mode-specific prompt (Extract / Analyze / Compare).
6. **Verify** — [`verify.py`](verify.py) traces every figure back to the sources, deterministically, and renders a "✓ N traced / ⚠ unverified" badge.
7. **Learn** — a second pass extracts facts into `MetricFact`, so the next similar question is faster and more deterministic.

---

## Tech stack

| Layer          | Choice                                                                              |
| -------------- | ----------------------------------------------------------------------------------- |
| Chat LLM       | Swappable: OpenRouter (any OpenAI-compatible model) or Ollama (`minimax-m3:cloud`)  |
| Embeddings     | `nomic-embed-text` (768-dim) via Ollama, with `search_query:`/`search_document:` prefixes |
| Vector store   | ChromaDB (cosine), persisted to disk                                                |
| Keyword search | `rank-bm25` via LangChain `BM25Retriever`                                            |
| Fusion         | LangChain `EnsembleRetriever` (Reciprocal Rank Fusion)                               |
| Reranker       | `BAAI/bge-reranker-base` cross-encoder (`sentence-transformers`)                     |
| PDF parsing    | PyMuPDF (text) · pdfplumber + Camelot + img2table (tables) · Tesseract (OCR)         |
| Vision         | `granite3.2-vision:2b` via Ollama — transcribes figures/diagrams                    |
| Fact cache     | Redis (L1) → SQLite `MetricFact` (L2) → RAG (L3)                                     |
| Backend        | Django + DRF, SQLite                                                                 |
| Frontend       | Plain HTML/CSS/JS + Plotly.js (no build step)                                        |

Everything except the chat LLM runs locally. Redis is optional — the cache silently falls back to SQLite if it's unreachable.

---

## Project layout

```
finrag/
├── data/<company>/*.pdf      # Source PDFs, grouped by company folder
├── packs/finance-india/      # Domain pack: companies, fiscal calendar, metrics, prompts
├── vectorstore/              # Persisted Chroma index (gitignored)
├── finrag_backend/           # Django project (settings, urls)
├── chat/                     # DRF app: models, views, RAG bridge
├── frontend/                 # index.html · style.css · app.js
├── evals/                    # qa.yaml (test cases) + run.py (eval runner)
├── query.py                  # Hybrid retrieval + intent + RAG chain + cache hook
├── nlu.py                    # LLM slot extraction (+ regex fallback)
├── reranker.py               # Cross-encoder reranker (stage 2 of retrieval)
├── verify.py                 # Deterministic figure-tracing (answer → sources)
├── facts.py / backfill.py    # Fact extraction + corpus pre-population
├── dashboard.py              # SQL → chart specs (no LLM)
├── cache.py                  # Redis L1 + SQLite L2 fact cache
├── llm_provider.py           # Provider factory: OpenRouter | Ollama
├── ingest.py / uploads.py    # Corpus ingest · per-chat upload ingest
└── config.py                 # Paths, model names, retrieval + cache params
```

---

## Evaluation & CI

The eval suite ([`evals/qa.yaml`](evals/qa.yaml), 30 cases) scores the system at three levels, and CI runs the deterministic slice on every push.

```bash
python evals/run.py --nlu            # query understanding only — no LLM, no index (CI runs this)
python evals/run.py --retrieval      # + retrieval & filter scoring (needs the index)
python evals/run.py                  # full run, generates answers (needs LLM + Ollama)
python evals/run.py --faithfulness   # groundedness / hallucination rate over the set
```

- **`--nlu`** — checks company/period/currency/numeric detection against the live parsers. Deterministic, runs in seconds, **no Ollama or vector index needed**. This is the [GitHub Actions](.github/workflows/ci.yml) gate: every push must keep it green.
- **`--faithfulness`** — generates real answers and reuses `verify.py`'s tracer to report a **groundedness score** (figures traceable to a source) and **hallucination rate**. On the current 30-case set this sits at **~91% grounded**, and the misses concentrate on RIIL figures — which have no `MetricFact` backfill yet, so they can only be traced against retrieved chunks. Backfilling RIIL should push this higher.

Why a split? Retrieval and answer quality need a built index + a running LLM, which a CI runner doesn't have. The understanding layer is pure and deterministic — so that's what guards every commit, while the heavier metrics are run locally.

---

## Local setup (without Docker)

```bash
# System deps (macOS)
brew install tesseract ghostscript redis
brew services start redis                    # optional; cache falls back to SQLite

# Models (Ollama)
ollama pull nomic-embed-text                 # embeddings — required
ollama pull granite3.2-vision:2b             # optional — only for figure transcription

# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate

# Chat LLM — set in .env (or shell):
#   OpenRouter (default):  OPENROUTER_API_KEY=sk-or-v1-...
#   Ollama (local):        LLM_PROVIDER=ollama

# Build the index from your PDFs, then run
rm -rf vectorstore && python ingest.py
python ingest_figures.py                     # optional: transcribe figures
python manage.py runserver                   # http://localhost:8000
```

CLI shortcut (no server): `python query.py "your question"`.

---

## Configuration

All settings live in [`config.py`](config.py) and read from environment / `.env`. The ones you're most likely to touch:

| Setting              | Default                     | What it controls                                   |
| -------------------- | --------------------------- | -------------------------------------------------- |
| `LLM_PROVIDER`       | `openrouter`                | `openrouter` (cloud) or `ollama` (local)           |
| `OPENROUTER_MODEL`   | a free model slug           | Which OpenRouter model answers                     |
| `DOMAIN_PACK`        | `finance-india`             | Which `packs/` config (companies, fiscal calendar) |
| `TOP_K`              | `8` (modes bump to 12)      | Chunks fed to the LLM                              |
| `RERANKER_ENABLED`   | `True`                      | Toggle the stage-2 cross-encoder                   |
| `REDIS_URL`          | `redis://localhost:6379/0`  | `""` to disable Redis (use SQLite only)            |
| `FACT_CACHE_TTL_SEC` | `86400` (24h)               | How long cached facts stay hot in Redis            |

> Schema-affecting changes (embedding model, chunk size) require wiping `vectorstore/` and re-ingesting.

---

## API

Base path `/api`. Highlights:

| Method | Path                      | Returns                                                           |
| ------ | ------------------------- | ---------------------------------------------------------------- |
| POST   | `/chats/{id}/messages`    | `{user_message, assistant_message, recall, chart, verification}` |
| POST   | `/chats/{id}/uploads`     | Indexes an uploaded PDF into its own Chroma collection           |
| GET    | `/dashboard?company=<slug>` | Chart-ready series from `MetricFact` (SQL, no RAG)             |
| GET    | `/recall?question=...`    | Related past analyses (pre-submit lookup)                        |

Full table and models (`Chat`, `Message`, `MetricFact`, `AnalysisNote`, `UploadedDoc`) are in [`finrag_backend/urls.py`](finrag_backend/urls.py) and [`chat/models.py`](chat/models.py).

---

## Design decisions (the interesting bits)

- **Two-stage retrieval.** Cheap hybrid search casts a wide net; an expensive cross-encoder rescores only the shortlist. Biggest accuracy lift after hybrid itself.
- **Deterministic verification, not an LLM judge.** Figures are checked by number-matching against the retrieved context — zero extra LLM calls, no judge to hallucinate. The model does the fuzzy writing; cheap code does the checkable checking.
- **Authoritative cached facts.** Validated values are injected as a synthetic `[CACHED-FACTS]` chunk the model must quote verbatim, eliminating re-reading errors on cells we've already confirmed.
- **Currency-aware retrieval.** A per-chunk INR/USD classifier drops USD duplicates when the INR twin exists, so the model can't misread `US$1,640M` as `₹1,640 cr`.
- **Period metadata at ingest time.** Filenames (`q1-2024.pdf` → `Q1FY24`) are parsed into hard retrieval filters, not left to the LLM to infer.
- **Fault tolerance throughout.** Cache, extractor, recall, reranker, verify — every layer is wrapped so the worst case is "slower RAG," never a broken answer.

---

## Known limitations

- **Single corpus collection.** Curated PDFs share one Chroma collection — fine at hundreds of PDFs; you'd want per-domain collections beyond that. (Uploads are already isolated.)
- **BM25 index is in RAM.** Needs rework at tens of thousands of chunks.
- **Cloud LLM isn't strictly offline.** The default ships context to OpenRouter/Ollama Cloud; switch to a local Ollama model for fully offline use.
- **Cached facts are authoritative.** If a source PDF is updated, stale facts persist until the cache TTL expires or you wipe them.
- **Reranker downloads ~280 MB** on first use (cached afterwards).
