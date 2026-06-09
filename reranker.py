"""Cross-encoder reranker — stage 2 of retrieval.

Stage 1 (hybrid BM25 + vector) is fast but shallow: it embeds the question
once and each chunk once, then compares vectors. Nuance like "in USD" vs
"(in US $ millions)" caption match often falls just below the cutoff.

Stage 2 (this module) feeds `[question + chunk]` into a cross-encoder so
the model attends to both jointly. ~100x slower per pair than cosine, but
we only ever call it on a pre-filtered ~50-candidate shortlist, so the
total cost is sub-second.

Drop-in, lazy-loaded, fail-open: if the model can't be imported or fails
to load on this machine, `rerank()` returns the input order unchanged so
retrieval still works. Toggle via `config.RERANKER_ENABLED`.
"""
from __future__ import annotations

import config


_MODEL = None
_LOAD_FAILED = False


def _get_model():
    """Lazy-load the cross-encoder. First call downloads the model
    (~280MB for bge-reranker-base) into the HuggingFace cache. Subsequent
    calls reuse the in-process instance.
    """
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    try:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(config.RERANKER_MODEL)
        print(f"  [reranker] loaded {config.RERANKER_MODEL}")
    except Exception as e:
        _LOAD_FAILED = True
        print(f"  [reranker] load failed: {type(e).__name__}: "
              f"{str(e)[:160]} -- falling back to retriever order")
    return _MODEL


def rerank(question: str, docs, top_k: int | None = None):
    """Re-score `docs` against `question` with the cross-encoder, return
    them sorted best-first. If `top_k` is given, truncate to that many.

    Fail-open: if reranking is disabled, the model isn't available, or the
    scoring call raises, we return the input order (truncated to `top_k`
    if provided) so the pipeline never breaks because of this stage.
    """
    if not config.RERANKER_ENABLED:
        return docs[:top_k] if top_k else list(docs)
    if not docs:
        return list(docs)
    # Single doc: nothing to rank.
    if len(docs) == 1:
        return list(docs)
    model = _get_model()
    if model is None:
        return docs[:top_k] if top_k else list(docs)
    try:
        pairs = [(question, d.page_content) for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
        out = [d for _, d in ranked]
    except Exception as e:
        print(f"  [reranker] predict failed: {type(e).__name__}: "
              f"{str(e)[:160]} -- using original order")
        return docs[:top_k] if top_k else list(docs)
    return out[:top_k] if top_k else out
