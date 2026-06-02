from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

LLM_MODEL = "gpt-oss:20b-cloud"
EMBEDDING_MODEL = "nomic-embed-text"
VISION_MODEL = "granite3.2-vision:2b"   # ~2GB, document-focused; alternatives: "llama3.2-vision", "moondream"
VISION_DPI = 144                        # render resolution for vision-model calls
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
