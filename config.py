from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OCR
OCR_ENABLED = True
OCR_DPI = 300                # render resolution for scanned pages
OCR_MIN_CHARS = 30           # if get_text() returns fewer chars, treat page as scanned
OCR_LANG = "eng"

COLLECTION_NAME = "finrag"
TOP_K = 8
MMR_FETCH_K = 24
MMR_LAMBDA = 0.7
