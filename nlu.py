"""LLM-based structured slot extraction with a regex fallback.

Replaces the patchwork of regex detectors in query.py (company aliases, period
patterns, range expansion, free-standing quarter) with ONE constrained-JSON
call. The schema mirrors what the retriever already consumes:

    {
      "companies":         [<slug>, ...],          # whitelisted to known
      "quarters":          [1..4, ...],            # may be empty
      "fys":               [<two-digit fy>, ...],  # may be empty
      "metrics":           [<free text>, ...],     # informational, not filtered
      "statement_variant": "standalone" | "consolidated" | null,
      "intent":            "lookup" | "trend" | "compare" | "explain"
    }

Periods are DERIVED from (quarters, fys) by `slots_to_periods` so the retriever
sees the same label set it does today (`Q3FY24`, `FY25`, ...).

Failure modes (timeout, malformed JSON, empty slots, validation wipes
everything) return None so the caller falls back to the existing regex path.
We never lose coverage, we only gain it.
"""

import json
from functools import lru_cache

from langchain_ollama import ChatOllama

import config
from parsers import PARSERS


# --- corpus whitelist (validation) -----------------------------------------

@lru_cache(maxsize=1)
def known_companies():
    """Valid company slugs — exactly what the ingest pipeline tagged onto chunks."""
    return set(PARSERS.keys())


@lru_cache(maxsize=1)
def known_fys():
    """FYs actually present in data/ — scanned from filenames so we don't
    accept hallucinated years (e.g. 'FY30')."""
    fys = set()
    for company in PARSERS:
        folder = config.DATA_DIR / company
        if not folder.exists():
            continue
        for pdf in folder.glob("*.pdf"):
            from parsers import parse_filename  # local import: avoid cycles
            meta = parse_filename(company, pdf.name) or {}
            fy = meta.get("fy")
            if isinstance(fy, int):
                fys.add(fy)
    return fys


# --- prompt -----------------------------------------------------------------

_EXTRACTION_SYSTEM = """You extract structured query slots from a user's question about financial documents.

Output ONLY a JSON object with these keys:
  "companies":         array of company slugs from {companies_list}. Empty if none mentioned.
  "quarters":          array of integers 1-4. Empty if no quarter is named.
  "fys":               array of two-digit fiscal years. EXPAND ranges. "FY21 through FY26" -> [21,22,23,24,25,26]. "last 3 fiscals" relative to FY26 -> [24,25,26]. Empty if none.
  "metrics":           array of short metric names mentioned ("revenue", "operating profit", "EPS", "net profit"). Empty if none.
  "statement_variant": "standalone", "consolidated", or null.
  "intent":            one of "lookup" (single value), "trend" (over time), "compare" (across companies/periods), "explain" (qualitative).

Rules:
- A free-standing "Q3" with an annual range ("Q3 ... from FY21 through FY26") means quarters=[3] AND fys=[21..26]. Do NOT drop the quarter.
- Use Indian FY convention: FY26 = Apr 2025 to Mar 2026. "fiscal 2024" -> 24.
- "last N quarters/years" -> resolve relative to the LATEST FY in the corpus ({latest_fy}).
- "infy"/"infosys" -> "infosys". "reliance"/"riil"/"reliance industrial infrastructure" -> "riil".
- If the question is non-financial (chitchat, document description), return all-empty slots with intent="explain".
- Output JSON only, no prose, no markdown."""


def _build_system_prompt():
    companies = sorted(known_companies())
    fys = sorted(known_fys())
    latest_fy = max(fys) if fys else 26
    return _EXTRACTION_SYSTEM.format(
        companies_list=companies,
        latest_fy=latest_fy,
    )


# --- extraction -------------------------------------------------------------

_EMPTY = {
    "companies": [], "quarters": [], "fys": [],
    "metrics": [], "statement_variant": None, "intent": "lookup",
}


def _coerce(slots):
    """Best-effort normalize whatever the LLM emitted into our schema. Anything
    unrecoverable raises and the caller returns None (-> regex fallback)."""
    out = dict(_EMPTY)
    out["companies"] = [str(c).lower().strip() for c in slots.get("companies") or []]
    out["quarters"] = [int(q) for q in slots.get("quarters") or [] if int(q) in (1, 2, 3, 4)]
    fys_raw = slots.get("fys") or []
    fys = []
    for y in fys_raw:
        y = int(y)
        if y >= 100:           # "2024" -> 24
            y = y % 100
        fys.append(y)
    out["fys"] = fys
    out["metrics"] = [str(m).strip() for m in slots.get("metrics") or [] if str(m).strip()]
    sv = slots.get("statement_variant")
    out["statement_variant"] = sv.lower() if isinstance(sv, str) and sv.lower() in ("standalone", "consolidated") else None
    intent = (slots.get("intent") or "lookup").lower()
    out["intent"] = intent if intent in ("lookup", "trend", "compare", "explain") else "lookup"
    return out


def _validate(slots):
    """Drop anything not in the corpus whitelist. Stops hallucinated slots from
    poisoning retrieval (e.g. 'tcs', 'FY30'). Returns the trimmed dict, or None
    if EVERYTHING got dropped (-> regex fallback gets a chance)."""
    company_wl = known_companies()
    fy_wl = known_fys()
    slots["companies"] = [c for c in slots["companies"] if c in company_wl]
    slots["fys"] = [y for y in slots["fys"] if y in fy_wl]
    # All-empty filter slots = nothing actionable; let regex try.
    if not slots["companies"] and not slots["fys"] and not slots["quarters"]:
        return None
    return slots


def slots_to_periods(slots):
    """Cross quarters × fys into canonical period labels the retriever filters on.

      quarters=[3], fys=[21,22,23] -> {"Q3FY21","Q3FY22","Q3FY23"}
      quarters=[],  fys=[24,25]    -> {"FY24","FY25"}
      quarters=[3], fys=[]         -> set()  (no period filter; quarter alone
                                              isn't a valid Chroma filter
                                              under our metadata schema)
    """
    quarters = slots.get("quarters") or []
    fys = slots.get("fys") or []
    if quarters and fys:
        return {f"Q{q}FY{fy:02d}" for q in quarters for fy in fys}
    if fys:
        return {f"FY{fy:02d}" for fy in fys}
    return set()


# --- public entry -----------------------------------------------------------

_LLM = None


def _get_llm():
    """Lazy ChatOllama in JSON mode. Temperature 0 so extraction is stable."""
    global _LLM
    if _LLM is None:
        _LLM = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
        )
    return _LLM


def extract_slots(question, history=None, llm=None):
    """Run the structured-extraction call. Returns the validated slot dict, or
    None on ANY failure (timeout, bad JSON, empty after validation). The caller
    must treat None as 'fall back to regex'."""
    if not question or not question.strip():
        return None
    llm = llm or _get_llm()
    try:
        msg = llm.invoke([
            ("system", _build_system_prompt()),
            ("human", question.strip()),
        ])
        content = getattr(msg, "content", None) or ""
        raw = json.loads(content)
        slots = _coerce(raw)
        return _validate(slots)
    except Exception as e:
        # Includes JSONDecodeError, connection errors, schema mismatches.
        print(f"  [nlu] extraction failed ({type(e).__name__}); regex fallback")
        return None
