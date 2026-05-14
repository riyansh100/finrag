from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

COLLECTION_NAME = "finrag"
TOP_K = 8
MMR_FETCH_K = 24
MMR_LAMBDA = 0.7
