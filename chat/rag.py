"""Bridge between the Django app and the RAG pipeline (query.ask()).

Keeps three concerns out of the view:
  - building the chat LLM exactly once (singleton, replacing Streamlit's
    @st.cache_resource),
  - turning DB Message rows into the history shape ask() expects,
  - turning ask()'s result (LangChain Documents + diagnostics) into plain
    JSON-able dicts for storage and the API response.
"""

import config
from langchain_ollama import ChatOllama

from query import ask as _ask

_llm = None


def get_llm():
    """One ChatOllama instance for the whole process."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        )
    return _llm


def _doc_to_dict(doc):
    """LangChain Document -> the JSON shape the frontend Sources panel needs."""
    meta = doc.metadata or {}
    return {
        "source": meta.get("source"),
        "page": meta.get("page"),
        "type": meta.get("type", "text"),
        "content": doc.page_content,
    }


def _build_flags(result):
    """Mirror the Streamlit flag captions, from ask()'s result dict."""
    flags = []
    if result.get("rewritten_query"):
        flags.append(f"Rewrote → {result['rewritten_query']}")
    if result.get("filtered_to"):
        flags.append(f"Filtered to: {result['filtered_to']}")
    if result.get("multi_sources"):
        flags.append(
            "Multi-author → per-author retrieval bonus: "
            + ", ".join(result["multi_sources"])
        )
    if result.get("numeric"):
        flags.append("Numeric intent → table-biased retrieval")
    return flags


def run_query(question, history):
    """Call the RAG pipeline and return JSON-able pieces.

    history: list of {"role", "content"} for THIS chat, already trimmed and
             NOT including the current question.
    Returns: {"answer", "sources", "flags", "rewritten_query"}.
    """
    result = _ask(question, history=history, llm=get_llm())
    return {
        "answer": result["answer"],
        "sources": [_doc_to_dict(d) for d in result["sources"]],
        "flags": _build_flags(result),
        "rewritten_query": result.get("rewritten_query"),
    }
