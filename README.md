# FinRAG

Local-first RAG chatbot for financial PDFs (quarterly reports, annual filings). Hybrid retrieval (BM25 + cosine) over a Chroma index, table-aware chunking, OCR + vision fallbacks, period/company-aware filtering, and a Extract/Analyze/Compare mode registry. Django + DRF backend, vanilla JS frontend.

On top of RAG sits a **three-layer analytics stack** (Redis cache → SQLite facts → RAG) that turns every answer into structured rows the bot reuses on future questions.

---

## Stack

| Layer            | Choice                                                                              |
| ---------------- | ----------------------------------------------------------------------------------- |
| LLM              | `gpt-oss:20b-cloud` via Ollama Cloud (any local Ollama chat model works)            |
| Embeddings       | `nomic-embed-text` (768-dim) with `search_query:` / `search_document:` prefixes     |
| Vector store     | ChromaDB (cosine), persisted to disk                                                |
| Keyword search   | `rank-bm25` via LangChain `BM25Retriever`                                           |
| Fusion           | LangChain `EnsembleRetriever` (RRF)                                                 |
| PDF parsing      | PyMuPDF (text) · pdfplumber + Camelot stream + img2table (tables) · Tesseract (OCR) |
| Vision           | `granite3.2-vision:2b` via Ollama — figure/diagram transcription                    |
| Memory (chat)    | LLM rewriter + last `HISTORY_TURNS` messages                                        |
| Memory (facts)   | Redis (L1) → SQLite `MetricFact` (L2) → RAG (L3)                                    |
| Backend          | Django + DRF, SQLite (`Chat`, `Message`, `MetricFact`, `AnalysisNote`)              |
| Frontend         | Plain HTML/CSS/JS, `marked.min.js` vendored                                         |

Everything except the chat LLM runs locally. Redis is optional — the cache silently falls back to SQLite if it's unreachable.

---

## Repository layout

```
finrag/
├── data/<company>/*.pdf      # Source PDFs, grouped by company folder
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
├── query.py                  # Hybrid retrieval + intent + RAG chain + cache hook
├── modes.py                  # Mode registry: Extract / Analyze / Compare
├── nlu.py                    # LLM-based structured slot extraction (with regex fallback)
├── facts.py                  # Post-answer fact extractor + persistence
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
  ├─ PROMPT[mode] | ChatOllama           # mode = extract | analyze | compare
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
| `LLM_MODEL`                      | `gpt-oss:20b-cloud`              |
| `EMBEDDING_MODEL`                | `nomic-embed-text`               |
| `CHUNK_SIZE` / `OVERLAP`         | `1000` / `200`                   |
| `TOP_K`                          | `8` (modes can bump to 12)       |
| `MAX_CONTEXT_CHUNKS`             | `24`                             |
| `HYBRID_BM25_WEIGHT`             | `0.5`                            |
| `HISTORY_TURNS`                  | `6`                              |
| `REDIS_URL`                      | `redis://localhost:6379/0` (`""` to disable) |
| `FACT_CACHE_TTL_SEC`             | `86400` (24h)                    |
| `FACT_CACHE_SHORTCIRCUIT_RAG`    | `True`                           |

Schema-affecting changes (embedding model, chunk size, ingest headers) require wiping `vectorstore/` and re-ingesting.

---

## API (`/api`)

| Method | Path                       | Body                       | Returns                                         |
| ------ | -------------------------- | -------------------------- | ----------------------------------------------- |
| GET    | `/modes`                   | —                          | `{modes, default}`                              |
| GET    | `/chats`                   | —                          | List of chats                                   |
| POST   | `/chats`                   | `{title?}`                 | New chat                                        |
| GET    | `/chats/{id}`              | —                          | Chat + full message list                        |
| DELETE | `/chats/{id}`              | —                          | 204                                             |
| POST   | `/chats/{id}/messages`     | `{question, mode?}`        | `{user_message, assistant_message, recall, rewritten_query}` |
| GET    | `/recall?question=...`     | —                          | `{recall, scope}` — pre-submit lookup           |
| GET    | `/notes/{id}`              | —                          | Full `AnalysisNote` body                        |

Models: `Chat`, `Message`, `MetricFact`, `FactProvenance`, `AnalysisNote`.

---

## Modes (`modes.py`)

| Mode      | top_k | Output shape                                                                 |
| --------- | ----- | ---------------------------------------------------------------------------- |
| `extract` | 8     | Verbatim figures with `[filename p.N]` citations. Terse.                     |
| `analyze` | 12    | `## Headline / ## Key observations / ## Risks & flags / ## Bottom line`      |
| `compare` | 12    | `## Framing / ## Comparison table / ## Deltas & interpretation / ## Bottom line` |

Mode is per-message and user-picked. Base prompt enforces Indian FY mapping (Q1=Apr–Jun … Q3=Oct–Dec), currency discipline (never mix ₹ with $), refuse rather than fabricate, and treat `[CACHED-FACTS]` chunks as authoritative.

---

## Prerequisites

```bash
brew install tesseract ghostscript redis    # macOS
brew services start redis                    # optional but recommended
ollama pull nomic-embed-text
ollama pull granite3.2-vision:2b             # optional, figures only
ollama signin                                # gpt-oss:20b-cloud (default)
# OR fully offline:
ollama pull llama3.1:8b && # set LLM_MODEL in config.py

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

## Design decisions

- **Hybrid BM25 + vector + RRF.** Biggest single quality lever; BM25 nails entity / line-item queries pure cosine misses.
- **Tables as first-class chunks** in Markdown, never split. Reserved seats in the context window for numeric questions.
- **Period metadata at ingest time** (filename → `Q3FY24`). Hard-filtered at retrieval, not left to the LLM.
- **LLM-based slot extraction** with regex fallback. Handles paraphrases ("infy topline"), relative dates ("last 3 fiscals"), and carry-over from history. Regex still wins on cost when the question is unambiguous.
- **Three-tier fact cache.** Redis → SQLite → RAG, with full-coverage short-circuit. Compounding: every answer fills the cache, so the bot gets faster the more it's used.
- **Authoritative cached values.** Cached facts are injected as a synthetic `[CACHED-FACTS]` chunk the LLM is instructed to use verbatim — eliminates re-reading errors on cells we've already validated.
- **Currency-aware retrieval.** Per-chunk INR/USD classifier (markers → EPS-symbol regex → magnitude fallback). USD chunks dropped when their INR twin exists, so the LLM can't misread `US$1,640M` as `₹1,640 cr`.
- **Statement-target-aware anchor probes.** Balance-sheet / cash-flow / P&L queries each use their own keyword probe, with text chunks allowed to win when pdfplumber misses the table boundary.
- **Recall is structured Jaccard, not embeddings.** Scope is a typed tuple; structured match is sharper than vector similarity for this use case.
- **Fault tolerance throughout.** Cache, extractor, recall — every layer wrapped in try/except. Worst case is "slower RAG", never a broken answer.

---

## Known limitations

- **No cross-encoder re-ranker.** RRF only; `bge-reranker-base` could push the right chunk to rank 1.
- **Single-collection index.** Fine at hundreds of PDFs; per-domain collections beyond that.
- **BM25 corpus in RAM.** Rework needed at tens of thousands of chunks.
- **No Q4 data in the current Infosys corpus** — FY totals not derivable from Q1–Q3 alone.
- **Cloud LLM is not strictly offline.** Default ships context to Ollama Cloud; switch `LLM_MODEL` for fully local.
- **Cache values are authoritative.** If a source PDF is updated, stale facts won't auto-refresh until TTL expires or you wipe (`redis-cli --scan --pattern 'fact:*' | xargs redis-cli del` + re-ask).
- **On-the-fly PDF upload** deferred — ingest assumes `data/<company>/` layout.
