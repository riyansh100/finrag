"""Deterministic answer verification — trace every material figure in an LLM
answer back to a source, with ZERO model calls.

The model's answer is checked against two things we already have in hand:
  1. the retrieved context chunks (what the model was actually shown), and
  2. the MetricFact store (authoritative numbers for the question's company).

A figure that appears in neither is flagged UNVERIFIED. This kills the main
hallucination mode (a number copied wrong, or invented) for the price of some
string/number matching — no extra LLM usage.

Scope is deliberately conservative to avoid false alarms: we hard-check only
COMMA-GROUPED figures (e.g. 1,31,322 / 131,322 / 12,310) — the absolute
monetary values a model lifts from a table, and the most damaging when wrong.
Bare integers (years, quarters, page numbers) and percentages / derived ratios
are NOT hard-checked, because they're often legitimately computed.
"""

import re

# A comma-grouped figure, optional decimal tail. Matches Indian (1,31,322) and
# Western (131,322) grouping, and 12,310.50.
_FIGURE_RE = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?")
# Any number — used to harvest the allowed set from the context chunks.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_float(tok):
    try:
        return float(str(tok).replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _numbers_in(text):
    """Every number in a blob, normalised to a rounded float (commas stripped),
    so '1,31,322', '131,322' and '131322' all collapse to the same value."""
    out = set()
    for m in _NUM_RE.findall(text or ""):
        f = _to_float(m)
        if f is not None:
            out.add(round(f, 2))
    return out


def _source_text(s):
    """Pull chunk text from either a serialized dict or a LangChain Document."""
    if isinstance(s, dict):
        return s.get("content") or ""
    return getattr(s, "page_content", "") or ""


def _fact_values(slots):
    """Authoritative MetricFact values for the question's companies. Fault
    tolerant: returns an empty set if the ORM isn't available (e.g. a bare
    `python query.py` run with no Django)."""
    vals = set()
    try:
        from chat.models import MetricFact
        companies = (slots or {}).get("companies") or []
        qs = MetricFact.objects.all()
        if companies:
            qs = qs.filter(company__in=companies)
        for v in qs.values_list("value", flat=True)[:8000]:
            vals.add(round(float(v), 2))
    except Exception:
        pass
    return vals


def verify_answer(answer, sources, slots=None):
    """Trace the comma-grouped figures in `answer` to `sources` + MetricFact.

    Returns:
        {
          "status":     "clean" | "flagged" | "n/a",
          "checked":    <# distinct figures checked>,
          "traced":     <# found in a source>,
          "unverified": [<figure strings not found>],
        }
    `n/a` = nothing worth checking (no comma-grouped figures in the answer).
    """
    figures = _FIGURE_RE.findall(answer or "")
    if not figures:
        return {"status": "n/a", "checked": 0, "traced": 0, "unverified": []}

    allowed = set()
    for s in sources or []:
        allowed |= _numbers_in(_source_text(s))
    allowed |= _fact_values(slots)

    seen, traced, unverified = set(), 0, []
    for tok in figures:
        if tok in seen:
            continue
        seen.add(tok)
        f = _to_float(tok)
        if f is None:
            continue
        # Exact match on the rounded value covers comma-grouping AND trailing
        # zeros (15.7 vs 15.70 both round to 15.7).
        if round(f, 2) in allowed:
            traced += 1
        else:
            unverified.append(tok)

    return {
        "status": "flagged" if unverified else "clean",
        "checked": traced + len(unverified),
        "traced": traced,
        "unverified": unverified,
    }
