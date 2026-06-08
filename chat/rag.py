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
            timeout=config.LLM_REQUEST_TIMEOUT_SEC,
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
    # Slice-2 cache diagnostics.
    if result.get("cache_short_circuit"):
        flags.append(f"Cache → {result.get('cache_hits', 0)} fact(s) "
                     f"covered the question; RAG skipped")
    elif result.get("cache_hits"):
        flags.append(f"Cache → {result['cache_hits']} fact(s) "
                     f"augmented retrieval")
    if result.get("recall"):
        flags.append(f"Recall → {len(result['recall'])} prior analysis "
                     f"match(es)")
    if result.get("upload_ids"):
        flags.append(f"Uploads → {len(result['upload_ids'])} "
                     f"attached PDF(s) searched")
    return flags


def run_query(question, history, mode=None, upload_ids=None):
    """Call the RAG pipeline and return JSON-able pieces.

    history: list of {"role", "content"} for THIS chat, already trimmed and
             NOT including the current question.
    mode: optional mode id ("extract" | "analyze" | "compare"). When set, the
          shared cached LLM is NOT reused — ask() builds a per-mode LLM so
          temperature/prompt match the mode.
    upload_ids: optional list of UploadedDoc ids attached to this turn. Each
          one's per-upload Chroma collection is searched alongside the corpus.
    Returns: {"answer", "sources", "flags", "mode", "rewritten_query"}.
    """
    upload_ids = list(upload_ids or [])
    if mode:
        result = _ask(question, history=history, mode=mode,
                      upload_ids=upload_ids)
    else:
        result = _ask(question, history=history, llm=get_llm(),
                      upload_ids=upload_ids)
    return {
        "answer": result["answer"],
        "sources": [_doc_to_dict(d) for d in result["sources"]],
        "flags": _build_flags(result),
        "mode": result.get("mode"),
        "rewritten_query": result.get("rewritten_query"),
        # Slots are passed downstream to the fact-extractor / analytics layer.
        # Not surfaced to the frontend.
        "slots": result.get("slots") or {},
        # Slice-3: prior analyses with overlapping scope (already JSON-safe).
        "recall": result.get("recall") or [],
        # Slice-4: which uploaded PDFs (if any) were searched for this turn.
        "upload_ids": result.get("upload_ids") or [],
    }
