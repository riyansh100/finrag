from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
#VISION_MODEL = "moondream"            # for ad-hoc figure description; alternative: "llama3.2-vision"
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
