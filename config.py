from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

LLM_MODEL = "minimax-m3:cloud"
# Hard ceiling on a single LLM call. Cloud Ollama occasionally accepts a
# request and then stops streaming -- without this, the chat UI just blinks
# forever. Anything over this triggers the friendly "model unreachable"
# branch in query.ask(). 120s is comfortably above a normal 30-60s answer
# but well short of "user gives up and reloads the tab".
LLM_REQUEST_TIMEOUT_SEC = 120
EMBEDDING_MODEL = "nomic-embed-text"
VISION_MODEL = "granite3.2-vision:2b"   # ~2GB, document-focused; alternatives: "llama3.2-vision", "moondream"
VISION_DPI = 96                       # render resolution for vision-model calls
FIGURE_DESCRIPTIONS_ENABLED = False      # describe figures at ingest, store as type="figure" chunks
FIGURE_MIN_IMAGE_PX = 200               # skip pages whose largest embedded image is smaller than this (likely logos)
FIGURE_PROMPT = (
    "This page contains a figure, diagram, chart, or screenshot from a project report. "
    "Describe it thoroughly: name every visible component, label, axis, and arrow. "
    "If it is an architecture or flow diagram, explain how the components connect. "
    "If it is a chart, summarise the trend and key values. Be concrete and specific."
)
VISION_TIMEOUT_SEC = 120                # kill vision call if it takes longer than this

# Conversational memory
HISTORY_TURNS = 6                       # last N *messages* sent to LLM (6 = 3 Q&A pairs)
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OCR
OCR_ENABLED = True
OCR_DPI = 300                # render resolution for scanned pages
OCR_MIN_CHARS = 30           # if get_text() returns fewer chars, treat page as scanned
OCR_LANG = "eng"

# Table extraction
TABLE_EXTRACTION_ENABLED = True
TABLE_MIN_ROWS = 2           # ignore "tables" smaller than this (likely noise)
TABLE_MIN_COLS = 2
TABLE_STREAM_FALLBACK_ENABLED = True   # Camelot stream when pdfplumber finds nothing
TABLE_OCR_ENABLED = True               # img2table on rendered images of OCR'd pages

# Hybrid retrieval
HYBRID_BM25_WEIGHT = 0.5     # vector weight = 1 - HYBRID_BM25_WEIGHT

COLLECTION_NAME = "finrag"
TOP_K = 8
MMR_FETCH_K = 24
MMR_LAMBDA = 0.7
MAX_CONTEXT_CHUNKS = 24      # hard cap on chunks sent to the LLM per query

# Cross-encoder reranker. Stage 1 (hybrid BM25 + vector) pulls a wide candidate
# set; stage 2 scores each (question, chunk) jointly with a cross-encoder and
# keeps the best ones. Cross-encoders read the question and chunk together, so
# they handle nuance ("in USD" vs "(in US $ millions)" caption match) that
# bi-encoder cosine misses. Set RERANKER_ENABLED=False to bypass the model
# entirely (e.g. on a fresh box before it's downloaded).
RERANKER_ENABLED = True
RERANKER_MODEL = "BAAI/bge-reranker-base"   # ~280MB, multilingual, strong on tables
RERANKER_FETCH_K = 50        # candidates passed to the reranker per retrieve()
# After reranking we still respect MAX_CONTEXT_CHUNKS; this just controls how
# many candidates the cross-encoder sees.

# Analytics layer (Slice 2) -- MetricFact cache + Redis L1.
# Redis is OPTIONAL. If unreachable the cache transparently falls back to
# SQLite (chat.MetricFact). Set REDIS_URL="" to disable Redis entirely.
REDIS_URL = "redis://localhost:6379/0"
FACT_CACHE_TTL_SEC = 24 * 60 * 60   # 24h; PDFs don't change often
FACT_CACHE_ENABLED = True
# When the cache covers EVERY requested (company, period, metric) cell, skip
# RAG entirely -- the LLM gets only the synthetic cached-facts chunk and a
# tiny set of source pages for citation context. Saves a full vector search.
FACT_CACHE_SHORTCIRCUIT_RAG = True

# On-the-fly PDF uploads -----------------------------------------------------
# Uploaded PDFs land in UPLOAD_DIR/ and are indexed into a per-upload Chroma
# collection named UPLOAD_CHROMA_PREFIX + "{id}". The curated `finrag` corpus
# is never touched -- upload collections are merged in at retrieval time and
# dropped when the parent chat is deleted.
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_MAX_MB = 25                          # reject larger PDFs at the endpoint
UPLOAD_CHROMA_PREFIX = "upload_"
UPLOAD_TOP_K = 12                           # chunks pulled from each upload per question
# When the user attaches a PDF and asks a question, reserve at least this
# fraction of MAX_CONTEXT_CHUNKS for upload chunks so the curated corpus
# can't crowd them out (e.g. asking about "riyansh's project" while three
# annual reports are also in scope).
UPLOAD_CONTEXT_FRACTION = 0.66
# Run the vision model on uploaded PDFs even when FIGURE_DESCRIPTIONS_ENABLED
# is off globally. Uploads are focused docs the user explicitly chose, so
# spending a few extra seconds per image-bearing page to get an architecture
# diagram described is worth it. Requires the vision model to be pulled
# locally (see config.VISION_MODEL).
UPLOAD_FIGURE_DESCRIPTIONS = False    # vision model off for uploads -- text + tables only
