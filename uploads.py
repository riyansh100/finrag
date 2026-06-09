"""On-the-fly PDF ingestion for chat-scoped uploads.

Wraps the existing ingest.load_pdf pipeline (PyMuPDF text + pdfplumber/Camelot
tables + optional OCR + optional figure descriptions) and writes the resulting
chunks into a PER-UPLOAD ChromaDB collection (`upload_{id}`) that lives
alongside the main `finrag` collection but is searched independently.

Why a separate collection per upload?
  - The curated corpus stays clean -- no risk that a noisy one-off PDF leaks
    into BM25 / vector search for everyone.
  - Deletion is trivial: drop the collection + the row + the saved file.
  - Per-chat / per-message scoping at retrieval time is just "which
    collections do we open" instead of metadata filtering on a huge corpus.

Everything here is best-effort: if pdfplumber fails on one page we still keep
the text, if Chroma chokes mid-batch we mark the upload FAILED with the error
message and the chat keeps working without that PDF.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as _Doc
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import reranker
from embeddings import make_embeddings
from ingest import load_pdf
from parsers import detect_upload_meta


# Embedder is built once and shared across uploads (same as the main corpus
# uses one). Building it allocates an Ollama client; cheap to reuse.
_EMBEDDINGS = None


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = make_embeddings()
    return _EMBEDDINGS


def collection_name_for(upload_id: int) -> str:
    return f"{config.UPLOAD_CHROMA_PREFIX}{upload_id}"


def open_upload_collection(upload_id: int) -> Chroma:
    """Open (or create) the Chroma collection for one upload. Same persist
    directory as the main corpus -- they coexist as separate collections."""
    return Chroma(
        collection_name=collection_name_for(upload_id),
        embedding_function=_get_embeddings(),
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def sha256_of(file_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    return h.hexdigest()


def _save_to_disk(upload_id: int, filename: str, file_bytes: bytes) -> Path:
    """Persist the uploaded PDF under UPLOAD_DIR/<upload_id>/<filename>.
    The id-prefixed subdir avoids name collisions across uploads."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    subdir = config.UPLOAD_DIR / str(upload_id)
    subdir.mkdir(parents=True, exist_ok=True)
    # Strip any path components a malicious client might have included.
    safe_name = Path(filename).name or "upload.pdf"
    dest = subdir / safe_name
    dest.write_bytes(file_bytes)
    return dest


def ingest_pdf(upload, file_bytes: bytes) -> dict:
    """Ingest one uploaded PDF into its per-upload Chroma collection.

    `upload` is a freshly-created UploadedDoc row (status=pending). On return
    the row is updated to ready/failed in-place; the caller saves it.

    Returns a small counters dict for the response payload.
    """
    # Save raw bytes to disk first so we can re-index later if ever needed.
    stored = _save_to_disk(upload.pk, upload.filename, file_bytes)
    upload.stored_path = str(stored.relative_to(config.BASE_DIR))
    upload.status = upload.STATUS_INDEXING
    upload.save(update_fields=["stored_path", "status", "updated_at"])

    # Tag every chunk so retrieval can attribute it back to the upload and the
    # frontend can render an "uploaded" badge in the Sources panel.
    # Best-effort filename detection stamps a period (e.g. "Q1FY18") so the
    # LLM has a structured anchor instead of guessing the FY from raw text.
    detected = detect_upload_meta(upload.filename)
    doc_meta = {
        "upload_id":  upload.pk,
        "chat_id":    upload.chat_id,
        "company":    "",     # leave the curated-corpus company slug empty
        "period":     detected.get("period") or "",
        "quarter":    detected.get("quarter"),
        "fy":         detected.get("fy"),
        "doc_type":   detected.get("doc_type") or "",
        "is_upload":  True,
    }

    # Vision-model figure descriptions are off by default for the curated
    # corpus (too slow at scale), but on for uploads -- the user uploaded
    # this PDF specifically, so we trade a few seconds per image-bearing
    # page for the ability to answer "describe this diagram". load_pdf reads
    # config.FIGURE_DESCRIPTIONS_ENABLED at call time, so we toggle it under
    # try/finally and restore the original value regardless of outcome.
    _saved_fig_flag = config.FIGURE_DESCRIPTIONS_ENABLED
    if config.UPLOAD_FIGURE_DESCRIPTIONS:
        config.FIGURE_DESCRIPTIONS_ENABLED = True
    try:
        try:
            text_docs, table_docs, figure_docs, counts = load_pdf(
                Path(stored), doc_meta=doc_meta,
            )
        except Exception as e:
            upload.status = upload.STATUS_FAILED
            upload.error = f"load_pdf: {type(e).__name__}: {str(e)[:200]}"
            upload.save(update_fields=["status", "error", "updated_at"])
            return {"chunks": 0, "pages": 0, "error": upload.error}
    finally:
        config.FIGURE_DESCRIPTIONS_ENABLED = _saved_fig_flag

    pages = max((d.metadata.get("page", 0) for d in text_docs), default=0)
    upload.pages = pages

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs)
    chunks = text_chunks + table_docs + figure_docs

    if not chunks:
        upload.status = upload.STATUS_FAILED
        upload.error = "No extractable content (text/tables/figures all empty)."
        upload.save(update_fields=["status", "error", "updated_at"])
        return {"chunks": 0, "pages": pages, "error": upload.error}

    coll_name = collection_name_for(upload.pk)
    upload.collection_name = coll_name

    vs = open_upload_collection(upload.pk)
    # Batch-add so a transient Ollama hiccup mid-PDF doesn't lose everything
    # already embedded -- same pattern ingest.main() uses for the corpus.
    BATCH = 32
    written = 0
    last_err = ""
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        try:
            vs.add_documents(batch)
            written += len(batch)
        except Exception as e:
            last_err = f"batch {i}: {type(e).__name__}: {str(e)[:160]}"
            print(f"  [upload {upload.pk}] {last_err}")

    upload.chunk_count = written
    if written == 0:
        upload.status = upload.STATUS_FAILED
        upload.error = last_err or "All embedding batches failed."
    else:
        upload.status = upload.STATUS_READY
        # Partial-success: keep the row READY but stash the last error so it's
        # discoverable. The PDF is still usable for retrieval.
        if last_err:
            upload.error = f"partial: {last_err}"
    upload.save(update_fields=[
        "pages", "chunk_count", "collection_name", "status",
        "error", "updated_at",
    ])

    return {
        "chunks":   written,
        "pages":    pages,
        "tables":   len(table_docs),
        "figures":  len(figure_docs),
        "ocr":      counts.get("ocr", 0),
    }


def drop_upload(upload) -> None:
    """Tear down everything an upload owns: Chroma collection, stored file,
    parent dir. Idempotent -- safe to call on a row that never finished
    indexing. The DB row itself is deleted by the caller (or by cascade).

    Every step is wrapped independently so a failure on one (e.g. Chroma
    collection already gone) doesn't block the rest. This is the cleanup
    path that runs on Chat cascade-delete via the pre_delete signal -- it
    MUST tolerate partially-indexed / inconsistent rows."""
    coll = upload.collection_name or collection_name_for(upload.pk)
    # Use the lower-level chromadb client to avoid building an Ollama-backed
    # Chroma() instance just to drop a collection. Spinning up the embedding
    # function for a teardown is wasteful and a known crash surface (Ollama
    # not running -> 500). Falls back to the Chroma() path only if needed.
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
        try:
            client.delete_collection(coll)
        except Exception:
            # Collection may already be gone -- treat as success.
            pass
    except Exception as e:
        print(f"  [upload {upload.pk}] drop_collection (client) failed: "
              f"{type(e).__name__}: {str(e)[:120]}")

    if upload.stored_path:
        try:
            parent = (config.BASE_DIR / upload.stored_path).parent
            if parent.exists() and parent.is_dir():
                shutil.rmtree(parent, ignore_errors=True)
        except Exception as e:
            print(f"  [upload {upload.pk}] file cleanup failed: "
                  f"{type(e).__name__}: {str(e)[:120]}")


# --- retrieval helpers (called from query.py) -------------------------------

# Statement-anchor probes -- same idea as the corpus retriever, but used per
# upload. When the question is about a specific financial statement, the
# literal keyword-matching tends to lose to "tangentially related" pages
# (e.g. a ratio-analysis page beats the actual balance-sheet page for "balance
# sheet data" because the former says "current ratio" and the latter is mostly
# numbers). The anchor probe pulls the page that contains the statement's
# signature line items, guaranteeing it lands in context.
_UPLOAD_ANCHORS = {
    "balance sheet": (
        "Total assets Total equity Total liabilities "
        "Cash and cash equivalents Trade receivables "
        "Property plant and equipment shareholders equity"
    ),
    "profit and loss": (
        "Revenues Cost of sales Gross profit Operating profit "
        "Net profit Basic EPS earnings per share"
    ),
    "cash flow": (
        "Cash flow from operating activities investing "
        "financing activities net increase decrease in cash"
    ),
    "changes in equity": (
        "Statement of changes in equity share capital reserves"
    ),
}


def retrieve_from_uploads(question: str, upload_ids, k: int = None,
                          statement_targets=None):
    """Pull up to `k` chunks from each of the given upload collections, merged.

    statement_targets: optional iterable of statement keywords (e.g.
        {"balance sheet"}) detected from the question. For each one we run an
        extra anchor probe so the page containing the signature rows ALWAYS
        lands in context, even when literal keyword similarity favours a
        different page.

    Returns a flat list of LangChain Documents. Empty list if no uploads or
    every collection lookup fails -- never raises.
    """
    if not upload_ids:
        return []
    k = k or config.UPLOAD_TOP_K
    statement_targets = set(statement_targets or [])
    out = []
    for uid in upload_ids:
        try:
            vs = open_upload_collection(uid)
            try:
                count = vs._collection.count()
            except Exception:
                count = 0
            if count == 0:
                continue
            safe_k = min(k, count)
            # SIMILARITY, not MMR. MMR adds non-determinism (re-running the
            # same question can return a different chunk set when scores tie),
            # which produced the "ratio table wasn't there, then was" bug.
            # The user uploaded a focused doc -- we want the top-k by cosine
            # similarity, same every time.
            retriever = vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": safe_k},
            )
            base_hits = retriever.invoke(question)
            out.extend(base_hits)

            # Anchor probes per detected statement target. For each one we run
            # a SECOND similarity pass with a probe rich in that statement's
            # signature rows, then promote every chunk from the matched page
            # (table extraction often splits one balance sheet across multiple
            # chunks; pulling the whole page guarantees the totals row is in
            # context even if it's in a different sub-chunk than the line
            # items the question asked about).
            if statement_targets:
                # Pre-load all chunks of this collection once so we can do
                # cheap page-level lookups for the "whole page" promotion.
                try:
                    res = vs.get()
                    all_in_coll = [
                        (d, m or {})
                        for d, m in zip(res["documents"], res["metadatas"])
                    ]
                except Exception:
                    all_in_coll = []

                for stmt in statement_targets:
                    probe = _UPLOAD_ANCHORS.get(stmt)
                    if not probe:
                        continue
                    try:
                        anchor_hits = vs.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": min(4, count)},
                        ).invoke(probe)
                    except Exception:
                        anchor_hits = []
                    out.extend(anchor_hits)
                    # Promote every chunk from anchor pages so split tables
                    # are fully visible.
                    anchor_pages = {
                        (h.metadata.get("source"), h.metadata.get("page"))
                        for h in anchor_hits
                    }
                    if anchor_pages and all_in_coll:
                        for content, meta in all_in_coll:
                            key = (meta.get("source"), meta.get("page"))
                            if key in anchor_pages:
                                out.append(_Doc(content, meta))
        except Exception as e:
            print(f"  [upload {uid}] retrieve failed: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            continue
    # Stage 2: cross-encoder rerank across the merged upload candidates.
    # We cap to RERANKER_FETCH_K first (so a chatty upload doesn't blow
    # up the cross-encoder), then keep the top UPLOAD_TOP_K per the
    # caller's k argument. Fail-open inside reranker.rerank().
    if out:
        rerank_top = (k or config.UPLOAD_TOP_K) * max(1, len(upload_ids))
        out = reranker.rerank(question,
                              out[:config.RERANKER_FETCH_K],
                              top_k=rerank_top)
    return out
