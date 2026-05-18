"""Embedding + vectorstore helpers.

Centralised so ingest.py and query.py stay in sync on:
  - nomic-embed-text task prefixes (search_query: / search_document:)
  - Chroma collection metric (cosine, not the L2 default)
"""

from typing import ClassVar

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import config


class NomicTaskEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings wrapper that prepends nomic task prefixes.

    nomic-embed-text-v1 / v1.5 are trained with task prefixes; without them
    similarity quality is significantly degraded.
    https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
    """

    DOC_PREFIX: ClassVar[str] = "search_document: "
    QUERY_PREFIX: ClassVar[str] = "search_query: "

    def embed_documents(self, texts):
        return super().embed_documents([self.DOC_PREFIX + t for t in texts])

    def embed_query(self, text):
        return super().embed_query(self.QUERY_PREFIX + text)


def make_embeddings():
    """Return the embedder used for both indexing and querying."""
    cls = NomicTaskEmbeddings if "nomic" in config.EMBEDDING_MODEL.lower() else OllamaEmbeddings
    return cls(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)


def make_vectorstore(embeddings=None):
    """Open (or create) the Chroma collection with cosine similarity."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or make_embeddings(),
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
