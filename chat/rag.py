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
    if result.get("company_filter"):
        flags.append(f"Company → {result['company_filter']}")
    if result.get("companies"):
        flags.append("Multi-company → " + ", ".join(result["companies"]))
    if result.get("period_filter"):
        flags.append(f"Period → {result['period_filter']}")
    if result.get("periods"):
        flags.append("Period range → " + ", ".join(result["periods"]))
    if result.get("numeric"):
        flags.append("Numeric intent → table-biased retrieval")
    return flags


def run_query(question, history, mode=None):
    """Call the RAG pipeline and return JSON-able pieces.

    history: list of {"role", "content"} for THIS chat, already trimmed and
             NOT including the current question.
    mode: optional mode id ("extract" | "analyze" | "compare"). When set, the
          shared cached LLM is NOT reused — ask() builds a per-mode LLM so
          temperature/prompt match the mode.
    Returns: {"answer", "sources", "flags", "mode", "rewritten_query"}.
    """
    if mode:
        result = _ask(question, history=history, mode=mode)
    else:
        result = _ask(question, history=history, llm=get_llm())
    return {
        "answer": result["answer"],
        "sources": [_doc_to_dict(d) for d in result["sources"]],
        "flags": _build_flags(result),
        "mode": result.get("mode"),
        "rewritten_query": result.get("rewritten_query"),
    }
