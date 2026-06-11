"""Ingest-time fact backfill: populate MetricFact from the report chunks we
already have in Chroma, instead of waiting for query-time extraction.

The numbers are already stored as table/text chunks in the vector store; this
script reshapes them into the structured MetricFact table so the dashboard can
chart them. It REUSES the validated query-time pipeline (facts.extract_facts_
from_answer -> normalize_fact -> persist_facts) — we just feed it report chunk
text instead of an assistant answer, plus an explicit company/period hint so
the extractor never has to guess the scope.

Usage:
    python backfill.py --doc q1-2024.pdf            # dry run, one document
    python backfill.py --doc q1-2024.pdf --persist  # write to MetricFact
    python backfill.py --company infosys            # dry run, all infosys docs
    python backfill.py --all --persist              # full backfill

Dry run prints the facts that WOULD be written; nothing touches the DB until
--persist is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finrag_backend.settings")
import django  # noqa: E402

django.setup()

import config  # noqa: E402
import facts  # noqa: E402
from embeddings import make_vectorstore  # noqa: E402
from parsers import PARSERS, parse_filename  # noqa: E402


# Chunk types worth extracting from. Tables hold the statements; some balance
# sheets / cash flows bleed into text chunks when pdfplumber misses the table
# boundary, so for small quarterly reports we include text too. Large annual
# reports (RIIL) carry hundreds of narrative text chunks with near-zero fact
# density, so we restrict those to tables only (see _types_for).
_EXTRACT_TYPES = ("table", "text")

# Per-extraction-call character budget. Chunks for one document are split into
# batches under this size so a big annual report becomes several calls instead
# of one over-stuffed prompt that blows the context window. Sized so a single
# quarterly report fits in ONE call (minimax-m3 has a large context) — halves
# the LLM calls vs the old 24k budget, which matters under cloud usage limits.
_BATCH_CHAR_BUDGET = 48000

# Extraction is the only step that costs cloud quota. We cache each document's
# normalized facts to disk so re-runs (and --persist) never re-call the LLM.
# Delete a file here (or pass --refresh) to force re-extraction of that doc.
_CACHE_DIR = config.BASE_DIR / "backfill_cache"

# Above this many chunks, treat a document as "large" and extract from tables
# only (skip narrative text) to keep cost/noise down.
_LARGE_DOC_CHUNKS = 100


def _types_for(all_chunks, source):
    """table-only for large docs (annual reports), table+text for small ones."""
    n = sum(1 for _, m in all_chunks if m.get("source") == source)
    return ("table",) if n > _LARGE_DOC_CHUNKS else _EXTRACT_TYPES


def _all_chunks():
    """Every chunk (page_content + metadata) in the corpus vector store."""
    vs = make_vectorstore()
    got = vs.get(include=["documents", "metadatas"])
    return list(zip(got["documents"], got["metadatas"]))


def _docs_for(company=None, doc=None):
    """Resolve which (company, source_filename, period) triples to backfill."""
    triples = []
    for comp, folder_parsers in PARSERS.items():
        if company and comp != company:
            continue
        folder = config.DATA_DIR / comp
        if not folder.exists():
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            if doc and pdf.name != doc:
                continue
            meta = parse_filename(comp, pdf.name) or {}
            period = meta.get("period")
            if not period:
                print(f"  [backfill] skip {pdf.name}: no period parsed")
                continue
            triples.append((comp, pdf.name, period))
    return triples


def _chunks_for_source(all_chunks, source):
    """The extract-worthy chunks for one source filename, page-ordered."""
    keep_types = _types_for(all_chunks, source)
    out = []
    for text, meta in all_chunks:
        if meta.get("source") != source:
            continue
        if meta.get("type", "text") not in keep_types:
            continue
        out.append((text, meta))
    out.sort(key=lambda tm: (tm[1].get("page") or 0))
    return out


def _batch_chunks(chunks):
    """Split page-ordered chunks into batches under _BATCH_CHAR_BUDGET so each
    extraction call stays within the model's context. A single oversized chunk
    still goes out alone (we never split mid-chunk)."""
    batches, cur, cur_len = [], [], 0
    for text, meta in chunks:
        tlen = len(text or "")
        if cur and cur_len + tlen > _BATCH_CHAR_BUDGET:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append((text, meta))
        cur_len += tlen
    if cur:
        batches.append(cur)
    return batches


def _cache_path(source):
    return _CACHE_DIR / f"{source}.json"


def _load_cache(source):
    """Return cached normalized facts for a doc, or None if not cached.
    Decimal values are restored from their JSON string form."""
    p = _cache_path(source)
    if not p.exists():
        return None
    try:
        rows = json.loads(p.read_text())
    except Exception as e:
        print(f"  [backfill] cache read failed for {source} "
              f"({type(e).__name__}); will re-extract")
        return None
    for r in rows:
        r["value"] = Decimal(str(r["value"]))
    return rows


def _save_cache(source, rows):
    _CACHE_DIR.mkdir(exist_ok=True)
    serializable = [{**r, "value": str(r["value"])} for r in rows]
    _cache_path(source).write_text(json.dumps(serializable, indent=1))


def _sanity_filter(rows):
    """Drop rows that are structurally impossible regardless of source text.
    Currently: per-share EPS can never be in 'millions' (the extractor
    occasionally tags a USD EPS value as usd_million)."""
    out = []
    for r in rows:
        if r["metric_key"] in ("eps_basic", "eps_diluted") and \
                r["unit"] == "usd_million":
            continue
        out.append(r)
    return out


def backfill_document(all_chunks, company, source, period, persist=False,
                      refresh=False):
    """Extract -> normalize -> (optionally) persist facts for one report.

    Returns the list of normalized fact dicts (the rows that would be / were
    written). The extractor is handed an explicit company+period hint so every
    fact is correctly scoped without guessing from page dates."""
    # Cache hit: skip the LLM entirely (free, quota-safe, resumable).
    if not refresh:
        cached = _load_cache(source)
        if cached is not None:
            print(f"  [backfill] {source} ({period}): {len(cached)} facts "
                  f"(cached)")
            if persist and cached:
                print(f"            persisted: "
                      f"{facts.persist_facts(cached, message=None)}")
            return cached

    chunks = _chunks_for_source(all_chunks, source)
    if not chunks:
        print(f"  [backfill] {source}: no extractable chunks")
        return []

    whitelist = set(PARSERS.keys())
    allowed_docs = {source}
    hint = (
        f"All figures below are from {company}'s financial statements for "
        f"reporting period {period}. Tag every extracted fact with "
        f"company={company} and period={period}."
    )
    question = (
        f"Extract every financial metric (P&L, balance sheet, cash flow) for "
        f"{company} {period} from the statements below."
    )

    batches = _batch_chunks(chunks)
    raw_total = 0
    normalized = []
    for bi, batch in enumerate(batches):
        # Each chunk tagged with its page so the LLM fills source_page.
        body = "\n\n".join(f"[{source} p.{m.get('page')}]\n{t}" for t, m in batch)
        answer_text = f"{hint}\n\n{body}"
        sources = [{"source": source, "page": m.get("page")} for _, m in batch]
        raw = facts.extract_facts_from_answer(question, answer_text, sources,
                                              whitelist)
        raw_total += len(raw)
        for r in raw:
            # Force the scope we KNOW from the filename -- guard against the
            # extractor mislabelling company/period off stray page text.
            r["company"] = company
            r["period"] = period
            n = facts.normalize_fact(r, whitelist, allowed_docs)
            if n is not None:
                normalized.append(n)

    # Dedupe within the doc: same (metric, unit, variant) can recur across
    # batches; keep the first (page-ordered) occurrence.
    seen, deduped = set(), []
    for n in _sanity_filter(normalized):
        key = (n["metric_key"], n["unit"], n["statement_variant"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(n)

    nb = f" in {len(batches)} batch(es)" if len(batches) > 1 else ""
    print(f"  [backfill] {source} ({period}): {raw_total} raw -> "
          f"{len(deduped)} valid facts{nb}")
    # Only cache a non-empty result. An empty list almost always means the
    # extraction call failed (quota/network) rather than a doc with no facts,
    # and we don't want to memoize that failure — leave it to be retried.
    if deduped:
        _save_cache(source, deduped)
    else:
        print(f"            (0 facts — likely a failed call; not cached, "
              f"will retry on next run)")
    if persist and deduped:
        counters = facts.persist_facts(deduped, message=None)
        print(f"            persisted: {counters}")
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="single source filename, e.g. q1-2024.pdf")
    ap.add_argument("--company", help="limit to one company slug")
    ap.add_argument("--all", action="store_true", help="every corpus document")
    ap.add_argument("--persist", action="store_true",
                    help="write to MetricFact (default: dry run)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the disk cache and re-extract via the LLM")
    args = ap.parse_args()

    if not (args.doc or args.company or args.all):
        ap.error("pass --doc, --company, or --all")

    triples = _docs_for(company=args.company, doc=args.doc)
    if not triples:
        print("No matching documents.")
        sys.exit(1)

    print(f"{'PERSIST' if args.persist else 'DRY RUN'} — "
          f"{len(triples)} document(s)\n")
    all_chunks = _all_chunks()

    grand = []
    for company, source, period in triples:
        rows = backfill_document(all_chunks, company, source, period,
                                 persist=args.persist, refresh=args.refresh)
        grand.extend(rows)

    # Summary: a compact metric x period grid so quality is eyeballable.
    print("\n" + "=" * 60)
    by_metric = defaultdict(list)
    for r in grand:
        by_metric[r["metric_key"]].append((r["period"], r["value"], r["unit"]))
    for metric in sorted(by_metric):
        cells = sorted(by_metric[metric])
        preview = "  ".join(f"{p}={v}{('' if u=='inr_crore' else ' '+u)}"
                            for p, v, u in cells[:8])
        more = f"  (+{len(cells)-8} more)" if len(cells) > 8 else ""
        print(f"  {metric:24} {preview}{more}")
    print(f"\n  TOTAL: {len(grand)} facts across "
          f"{len({r['period'] for r in grand})} period(s)")
    print("=" * 60)
    if not args.persist:
        print("Dry run — nothing written. Re-run with --persist to commit.")


if __name__ == "__main__":
    main()
