"""Per-company filename parsers.

Each parser maps `(filename) -> dict` of metadata stamped onto every chunk of
that document. The ingest pipeline walks `data/<company>/*.pdf` and looks up
the parser by folder name.

Metadata schema (all chunks):
    company   str    canonical company slug ("infosys", "riil", ...)
    doc_type  str    "quarterly" | "annual"
    period    str    canonical period label, used by retrieval filters
                       quarterly -> "Q1FY26"
                       annual    -> "FY25"
    quarter   int | None    1..4 for quarterlies, None for annuals
    fy        int    fiscal year, last two digits (26 means FY26 = Apr 2025 - Mar 2026)

A "fiscal year" follows the Indian convention: FY26 runs Apr 2025 - Mar 2026.
For Infosys quarterly filenames like `q1-2026.pdf`, the trailing year IS the FY.
"""

import re


# --- per-company parsers ---------------------------------------------------

_INFOSYS_RE = re.compile(r"^q([1-4])-(\d{4})\.pdf$", re.IGNORECASE)


def parse_infosys_quarterly(filename):
    """`q1-2026.pdf` -> Q1FY26 metadata."""
    m = _INFOSYS_RE.match(filename)
    if not m:
        return None
    quarter = int(m.group(1))
    year = int(m.group(2))      # 2026
    fy = year % 100             # 26
    return {
        "company": "infosys",
        "doc_type": "quarterly",
        "period": f"Q{quarter}FY{fy:02d}",
        "quarter": quarter,
        "fy": fy,
    }


# `Annual-Report-2024-25.pdf` -- FY ends in the second year (Mar 2025 -> FY25).
_RIIL_ANNUAL_RE = re.compile(
    r"^annual[-_ ]report[-_ ](\d{4})[-_ ](\d{2,4})\.pdf$", re.IGNORECASE,
)


def parse_riil_annual(filename):
    m = _RIIL_ANNUAL_RE.match(filename)
    if not m:
        return None
    end_year = int(m.group(2))
    if end_year < 100:          # "2024-25" -> 25
        fy = end_year
    else:                       # "2024-2025" -> 25
        fy = end_year % 100
    return {
        "company": "riil",
        "doc_type": "annual",
        "period": f"FY{fy:02d}",
        "quarter": None,
        "fy": fy,
    }


# --- registry --------------------------------------------------------------

# Folder name (lowercased) -> list of parsers to try in order.
PARSERS = {
    "infosys": [parse_infosys_quarterly],
    "riil":    [parse_riil_annual],
}


def parse_filename(company_folder, filename):
    """Look up the company's parsers and return metadata, or None on no match.

    Falls back to a permissive {"company": <folder>} stub so an unrecognised
    filename in a known folder still gets the company tag (and we can debug it
    from the ingest log).
    """
    parsers = PARSERS.get(company_folder.lower(), [])
    for fn in parsers:
        meta = fn(filename)
        if meta is not None:
            return meta
    # Unknown filename shape inside a known company folder.
    if company_folder.lower() in PARSERS:
        return {
            "company": company_folder.lower(),
            "doc_type": "unknown",
            "period": None,
            "quarter": None,
            "fy": None,
        }
    return None
