"""Incremental, restartable vision-pass that adds figure descriptions to the
existing Chroma collection without touching anything else.

Design:
  - One vision call per figure-bearing native page (skips OCR'd pages).
  - Skips pages already described (idempotent — safe to rerun).
  - Per-call timeout via the ollama HTTP client so a wedged model doesn't
    block the whole script.
  - Writes after every PDF, so a mid-run crash doesn't lose prior work.
  - Verbose per-page progress with elapsed time.

Usage:
    python ingest_figures.py                  # process all PDFs, all figure pages
    python ingest_figures.py --source riyansh # only PDFs whose name matches
    python ingest_figures.py --limit 3        # at most 3 vision calls (validation)
    python ingest_figures.py --dry-run        # count what would be done, no calls
    python ingest_figures.py --force          # re-describe even if already done
    python ingest_figures.py --prompt "..."   # override config.FIGURE_PROMPT
"""

import argparse
import base64
import sys
import time

import fitz
import ollama

import config
from embeddings import make_vectorstore
from ingest import (
    _chunk_header,
    _page_has_significant_image,
    _render_page_image,
    author_from_filename,
)
from langchain_core.documents import Document


def _ollama_client():
    return ollama.Client(host=config.OLLAMA_BASE_URL, timeout=config.VISION_TIMEOUT_SEC)


def _describe_page(client, page, prompt, model):
    """Single vision call. Returns description text or '' on failure."""
    _, png_bytes = _render_page_image(page, dpi=config.VISION_DPI)
    try:
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(png_bytes).decode("ascii")],
            }],
            options={"temperature": 0.1},
        )
        return (response.get("message", {}).get("content") or "").strip()
    except Exception as e:
        return f"__ERROR__ {type(e).__name__}: {e}"


def _existing_figure_pages(vs, filename):
    """Return set of page numbers already described for this filename."""
    res = vs.get(where={"$and": [{"source": filename}, {"type": "figure"}]})
    return {m.get("page") for m in (res.get("metadatas") or [])}


def _match_pdfs(query):
    paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not query:
        return paths
    q = query.lower()
    matches = [p for p in paths if q in p.name.lower()]
    if not matches:
        sys.exit(f"No PDFs match {query!r}. Available: {[p.name for p in paths]}")
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Only process PDFs whose filename contains this string")
    parser.add_argument("--limit", type=int, default=None, help="Max vision calls this run")
    parser.add_argument("--dry-run", action="store_true", help="Show plan, do not call vision")
    parser.add_argument("--force", action="store_true", help="Re-describe already-described pages")
    parser.add_argument("--prompt", default=config.FIGURE_PROMPT, help="Override the vision prompt")
    parser.add_argument("--model", default=config.VISION_MODEL, help="Override the vision model")
    args = parser.parse_args()

    vs = make_vectorstore()
    client = _ollama_client() if not args.dry_run else None
    pdfs = _match_pdfs(args.source)

    print(f"Vision model: {args.model}  |  timeout: {config.VISION_TIMEOUT_SEC}s/call  "
          f"|  DPI: {config.VISION_DPI}")
    if args.limit is not None:
        print(f"Max vision calls this run: {args.limit}")
    if args.force:
        print("Force mode: ignoring existing figure chunks.")
    if args.dry_run:
        print("Dry-run: counting only, no vision calls.\n")
    else:
        print()

    total_done = 0
    total_skipped = 0
    total_failed = 0

    for path in pdfs:
        author = author_from_filename(path.name)
        described = set() if args.force else _existing_figure_pages(vs, path.name)
        doc = fitz.open(path)

        plan = []  # list of (page_num, page)
        for i, page in enumerate(doc):
            page_num = i + 1
            if page_num in described:
                continue
            text = page.get_text().strip()
            used_ocr = len(text) < config.OCR_MIN_CHARS  # heuristic; mirrors ingest.py
            if used_ocr:
                continue
            if not _page_has_significant_image(page):
                continue
            plan.append((page_num, page))

        if not plan:
            print(f"  {path.name}: nothing to do "
                  f"({len(described)} already described)")
            doc.close()
            continue

        print(f"\n{path.name}: {len(plan)} page(s) to describe "
              f"({len(described)} already done)")

        new_chunks = []
        for n, (page_num, page) in enumerate(plan, 1):
            if args.limit is not None and total_done >= args.limit:
                print(f"  [limit reached: {args.limit}]")
                break
            tag = f"  [{n}/{len(plan)}] p.{page_num}"
            if args.dry_run:
                print(f"{tag}: (dry-run)")
                total_done += 1
                continue

            t0 = time.time()
            print(f"{tag}: ", end="", flush=True)
            desc = _describe_page(client, page, args.prompt, args.model)
            elapsed = time.time() - t0
            if desc.startswith("__ERROR__"):
                print(f"FAIL ({elapsed:.1f}s) — {desc[10:]}")
                total_failed += 1
                continue
            if not desc:
                print(f"empty response ({elapsed:.1f}s)")
                total_failed += 1
                continue

            new_chunks.append(Document(
                page_content=_chunk_header(path.name, author, page=page_num)
                             + "Figure description:\n" + desc,
                metadata={"source": path.name, "author": author,
                          "page": page_num, "type": "figure", "ocr": False,
                          "vision_model": args.model},
            ))
            total_done += 1
            preview = desc.replace("\n", " ")[:80]
            print(f"ok ({elapsed:.1f}s) — {preview}...")

        # persist after each PDF
        if new_chunks and not args.dry_run:
            vs.add_documents(new_chunks)
            print(f"  → persisted {len(new_chunks)} figure chunk(s) "
                  f"(collection now {vs._collection.count()} vectors).")

        doc.close()
        if args.limit is not None and total_done >= args.limit:
            break

    print()
    print("=" * 60)
    print(f"  described:  {total_done}")
    print(f"  failed:     {total_failed}")
    print(f"  skipped:    {total_skipped}  (pre-existing)")
    print("=" * 60)


if __name__ == "__main__":
    main()
