"""Filename -> period metadata, driven by the active domain pack.

The per-company filename rules, the company registry, and the fiscal-calendar
convention all live in packs/<DOMAIN_PACK>/pack.yaml now (see domain.py). This
module is a thin compatibility layer so the ingest / backfill / nlu / upload
call sites keep their existing imports.

Metadata schema (stamped onto every chunk):
    company   str          canonical company slug ("infosys", "riil", ...)
    doc_type  str          "quarterly" | "annual" | "unknown"
    period    str | None   canonical period label ("Q1FY26", "FY25")
    quarter   int | None   1..4 for quarterlies, None for annuals
    fy        int | None   two-digit fiscal year (26 == FY26)
"""

from domain import get_pack


# Backwards-compatible registry: {slug: [parser specs]}. Consumers only rely on
# `.keys()`, `in`, and `.items()` (to enumerate known companies), all of which
# this dict supports.
PARSERS = {slug: get_pack().parsers_for(slug) for slug in get_pack().company_slugs()}


def detect_upload_meta(filename: str) -> dict:
    """Best-effort metadata for an ad-hoc upload filename (no company folder).
    Returns a period stub without a company tag, or {} if nothing matches."""
    return get_pack().detect_upload_meta(filename)


def parse_filename(company_folder, filename):
    """Look up the company's parsers and return metadata, or None on no match.

    Falls back to a permissive {"company": <folder>} stub so an unrecognised
    filename in a known folder still gets the company tag."""
    return get_pack().parse_filename(company_folder, filename)
