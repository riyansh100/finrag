"""Post-answer fact extraction + persistence.

Second-pass LLM call that reads an assistant answer, the original question,
and the retrieved-source list, and emits structured facts in JSON. Facts are
normalised (metric -> canonical key, unit -> controlled vocabulary) and
upserted into the MetricFact table. Every write also goes to FactProvenance
for audit. The assistant answer itself is mirrored into AnalysisNote keyed by
scope so Slice 3 can surface recalls.

This module is intentionally fault-tolerant: any exception during extraction
is logged and swallowed. The user-facing answer flow MUST NOT break because
the analytics layer hiccuped.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Iterable

from langchain_ollama import ChatOllama

import config


# ---------------------------------------------------------------------------
# Canonical metric vocabulary.
#
# The extractor LLM is told to use these keys directly. As a safety net we
# also build a reverse lookup so free-text metric names ("topline",
# "operating margin") still resolve. Extending the vocabulary = adding one
# entry; no schema change.
# ---------------------------------------------------------------------------

CANONICAL_METRICS: dict[str, list[str]] = {
    # P&L
    "revenue":               ["revenue", "revenues", "revenue from operations",
                              "topline", "total revenue", "net revenue",
                              "total income"],
    "cost_of_sales":         ["cost of sales", "cost of goods sold", "cogs",
                              "cost of revenue"],
    "gross_profit":          ["gross profit"],
    "selling_marketing_exp": ["selling and marketing expenses",
                              "selling marketing expenses"],
    "general_admin_exp":     ["general and administration expenses",
                              "general and administrative expenses",
                              "g&a expenses", "admin expenses"],
    "total_operating_exp":   ["total operating expenses"],
    "operating_profit":      ["operating profit", "ebit", "operating income"],
    "operating_margin_pct":  ["operating margin", "op margin",
                              "operating profit margin"],
    "other_income":          ["other income"],
    "pbt":                   ["profit before tax", "pbt",
                              "profit before income taxes"],
    "tax":                   ["tax", "income tax expense", "tax expense"],
    "pat":                   ["profit after tax", "pat", "net profit",
                              "net income", "profit for the year",
                              "profit for the period"],
    "pat_margin_pct":        ["pat margin", "net margin", "net profit margin"],
    "eps_basic":             ["basic eps", "basic earnings per share"],
    "eps_diluted":           ["diluted eps", "diluted earnings per share"],

    # Balance sheet
    "total_assets":          ["total assets"],
    "total_equity":          ["total equity", "total shareholders equity",
                              "net worth", "shareholders funds"],
    "total_liabilities":     ["total liabilities"],
    "cash_and_equivalents":  ["cash and cash equivalents",
                              "cash and bank balances"],
    "current_investments":   ["current investments"],
    "non_current_investments": ["non current investments",
                                "non-current investments"],
    "trade_receivables":     ["trade receivables", "accounts receivable",
                              "debtors"],
    "ppe":                   ["property, plant and equipment", "ppe",
                              "fixed assets"],

    # Cash flow
    "cash_from_operations":  ["cash flow from operating activities",
                              "operating cash flow", "ocf",
                              "net cash from operating activities"],
    "cash_from_investing":   ["cash flow from investing activities",
                              "net cash from investing activities"],
    "cash_from_financing":   ["cash flow from financing activities",
                              "net cash from financing activities"],

    # Operational
    "headcount":             ["headcount", "employees", "employee count",
                              "total employees"],
    "dso_days":              ["dso", "day's sales outstanding",
                              "days sales outstanding"],
    "roe_pct":               ["roe", "return on equity"],
}


def _build_metric_lookup():
    rev = {}
    for canonical, aliases in CANONICAL_METRICS.items():
        rev[canonical.lower()] = canonical
        for alias in aliases:
            rev[alias.lower()] = canonical
    return rev


_METRIC_LOOKUP = _build_metric_lookup()


def canonicalize_metric(raw: str) -> str | None:
    """'topline' -> 'revenue'; 'Operating Margin %' -> 'operating_margin_pct';
    returns None if nothing reasonable matches."""
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw.strip().lower()).rstrip(":")
    if key in _METRIC_LOOKUP:
        return _METRIC_LOOKUP[key]
    # Tolerant: strip trailing "%" / units the model may append.
    key2 = re.sub(r"\s*(\(.*\)|%|crore|cr|lakh|million|m|usd|inr|rs\.?)\s*$",
                  "", key).strip()
    if key2 in _METRIC_LOOKUP:
        return _METRIC_LOOKUP[key2]
    return None


# ---------------------------------------------------------------------------
# Unit normalisation. Maps free-text units the LLM might emit to our
# controlled vocabulary (see chat.models.UNIT_CHOICES).
# ---------------------------------------------------------------------------

_UNIT_LOOKUP = {
    # INR crore family
    "inr_crore": "inr_crore", "crore": "inr_crore", "cr": "inr_crore",
    "crores": "inr_crore", "₹ crore": "inr_crore", "rs crore": "inr_crore",
    "rs. crore": "inr_crore", "₹crore": "inr_crore",
    # INR lakh
    "inr_lakh": "inr_lakh", "lakh": "inr_lakh", "lakhs": "inr_lakh",
    "₹ lakh": "inr_lakh",
    # USD million
    "usd_million": "usd_million", "us$ million": "usd_million",
    "us$ millions": "usd_million", "usd million": "usd_million",
    "$ million": "usd_million", "$m": "usd_million", "million usd": "usd_million",
    # Bare currencies (last resort)
    "inr": "inr", "rs": "inr", "rs.": "inr", "₹": "inr",
    "usd": "usd", "$": "usd",
    # Per-share / percent / count
    "pct": "pct", "%": "pct", "percent": "pct", "percentage": "pct",
    "rupees": "rupees", "₹/share": "rupees", "rs per share": "rupees",
    "count": "count", "people": "count", "employees": "count",
    "ratio": "ratio", "x": "ratio", "times": "ratio",
    "days": "days",
}


def canonicalize_unit(raw: str) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    return _UNIT_LOOKUP.get(key)


# ---------------------------------------------------------------------------
# Value parsing. The LLM might emit "38,821", "38821", "38821.5",
# "(2,542)" (negative), or even "38821.0". Decimal is safe.
# ---------------------------------------------------------------------------

def parse_value(raw) -> Decimal | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float, Decimal)):
        try:
            return Decimal(str(raw))
        except InvalidOperation:
            return None
    s = str(raw).strip().replace(",", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.lstrip("₹$ ").strip()
    if not s:
        return None
    try:
        v = Decimal(s)
    except InvalidOperation:
        return None
    return -v if neg else v


# ---------------------------------------------------------------------------
# Extractor LLM
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """You extract structured financial facts from an assistant's answer.

You will be given:
  - the user's question
  - the assistant's answer (markdown, may contain tables and inline citations)
  - the list of source documents available

Emit ONLY a JSON object of this exact shape:
{
  "facts": [
    {
      "company":         "<one of: {companies_list}>",
      "period":          "<canonical period label, e.g. Q3FY24 or FY25>",
      "metric":          "<canonical key from the metric list below, OR a clear English name>",
      "value":           <number, negative allowed, no thousands separators>,
      "unit":            "<one of: inr_crore, inr_lakh, usd_million, pct, rupees, count, ratio, days>",
      "statement_variant": "standalone" | "consolidated" | null,
      "source_doc":      "<filename from the source list>",
      "source_page":     <page number, integer>,
      "confidence":      <0.0 to 1.0>
    },
    ...
  ]
}

CANONICAL METRIC KEYS (prefer these):
{metrics_list}

Hard rules:
- ONLY emit facts the answer actually states. Do NOT compute deltas, margins, or YoY changes that aren't already in the answer text.
- ONE row per (company, period, metric, statement_variant, unit). No duplicates.
- If the answer says "n/a" or "not available" for a cell, OMIT that row -- do not emit a null-value fact.
- Period MUST be canonical: Q[1-4]FY[2-digit-year] or FY[2-digit-year]. Convert "quarter ended December 31, 2023" -> Q3FY24 using Indian FY (Apr-Mar).
- Unit MUST come from the allowed list. Map "₹ crore"/"crores"/"cr" -> inr_crore. "US$ millions" -> usd_million. "%" -> pct. "₹"/"Rs" per share -> rupees.
- source_doc must be a filename from the supplied source list; source_page must be an integer.
- If you cannot identify the company or period for a fact, OMIT it. Better to skip than guess.
- Output JSON only. No prose, no markdown, no backticks."""


def _build_extraction_prompt(companies_whitelist):
    # NOTE: cannot use str.format here — the prompt body contains literal
    # JSON braces that .format() would misread as placeholders. Plain
    # string replace is enough and stays robust to future prompt edits.
    return (_EXTRACTION_SYSTEM
            .replace("{companies_list}",
                     ", ".join(sorted(companies_whitelist)) or "<none>")
            .replace("{metrics_list}",
                     ", ".join(sorted(CANONICAL_METRICS.keys()))))


_EXTRACTOR_LLM = None


def _get_extractor_llm():
    """Lazy ChatOllama in JSON mode, temperature 0 (deterministic extraction)."""
    global _EXTRACTOR_LLM
    if _EXTRACTOR_LLM is None:
        _EXTRACTOR_LLM = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            timeout=config.LLM_REQUEST_TIMEOUT_SEC,
        )
    return _EXTRACTOR_LLM


def _format_sources(sources: Iterable[dict]) -> str:
    """Compact source list for the extractor prompt: one line per (doc, page)."""
    seen = set()
    lines = []
    for s in sources or []:
        key = (s.get("source"), s.get("page"))
        if key in seen or not key[0]:
            continue
        seen.add(key)
        lines.append(f"- {key[0]} p.{key[1]}")
    return "\n".join(lines) or "(no sources)"


def extract_facts_from_answer(
    question: str,
    answer: str,
    sources: list[dict],
    companies_whitelist: Iterable[str],
    llm=None,
) -> list[dict]:
    """Run the second-pass extractor. Returns a list of raw fact dicts (NOT
    yet normalised or persisted). Returns [] on any failure -- callers must
    treat empty as 'extractor produced nothing usable'."""
    if not answer or not answer.strip():
        return []
    llm = llm or _get_extractor_llm()
    sys_prompt = _build_extraction_prompt(companies_whitelist)
    user_msg = (
        f"USER QUESTION:\n{question.strip()}\n\n"
        f"ASSISTANT ANSWER:\n{answer.strip()}\n\n"
        f"SOURCE DOCUMENTS (filename + page):\n{_format_sources(sources)}"
    )
    try:
        resp = llm.invoke([
            ("system", sys_prompt),
            ("human", user_msg),
        ])
        content = getattr(resp, "content", None) or ""
        data = json.loads(content)
        facts = data.get("facts") or []
        if not isinstance(facts, list):
            return []
        return facts
    except Exception as e:
        print(f"  [facts] extraction failed ({type(e).__name__}: {str(e)[:120]})")
        return []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_PERIOD_RE = re.compile(r"^(Q[1-4])?FY\d{2}$")


def _normalize_period(raw: str) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().upper().replace(" ", "")
    # "Q3FY2024" -> "Q3FY24"
    m = re.match(r"^(Q[1-4])?FY(\d{2,4})$", s)
    if not m:
        return None
    q, y = m.group(1) or "", int(m.group(2))
    y = y % 100
    return f"{q}FY{y:02d}"


def _normalize_company(raw: str, whitelist: set[str]) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().lower()
    return s if s in whitelist else None


def normalize_fact(
    raw: dict, companies_whitelist: set[str], allowed_docs: set[str]
) -> dict | None:
    """Apply all canonicalisers + validation. Returns a dict ready for
    MetricFact persistence, or None if any required field is unrecoverable."""
    company = _normalize_company(raw.get("company"), companies_whitelist)
    period = _normalize_period(raw.get("period"))
    metric = canonicalize_metric(raw.get("metric") or "")
    value = parse_value(raw.get("value"))
    unit = canonicalize_unit(raw.get("unit") or "")
    if not (company and period and metric and value is not None and unit):
        return None
    source_doc = (raw.get("source_doc") or "").strip()
    if allowed_docs and source_doc and source_doc not in allowed_docs:
        # Hallucinated source -> drop it but keep the fact (provenance just
        # won't link to a page).
        source_doc = ""
    try:
        source_page = int(raw.get("source_page")) if raw.get("source_page") is not None else None
    except (TypeError, ValueError):
        source_page = None
    sv = (raw.get("statement_variant") or "").strip().lower()
    if sv not in ("standalone", "consolidated"):
        sv = ""
    try:
        conf = float(raw.get("confidence", 1.0))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 1.0
    return {
        "company": company,
        "period": period,
        "metric_key": metric,
        "value": value,
        "unit": unit,
        "statement_variant": sv,
        "source_doc": source_doc,
        "source_page": source_page,
        "confidence": conf,
    }


def _write_through_cache(fact_row) -> None:
    """Best-effort write-through to the Redis L1. Local import keeps facts.py
    importable without cache.py being loaded (and avoids a cycle)."""
    try:
        import cache
        cache.write_through(fact_row)
    except Exception as e:
        print(f"  [facts] cache write-through skipped ({type(e).__name__}: {str(e)[:80]})")


def persist_facts(normalized: list[dict], message=None) -> dict:
    """Upsert normalized facts into MetricFact and append to FactProvenance.

    Returns a counters dict: {"inserted", "overwritten", "duplicate", "skipped"}.
    Safe to call with []. Wrapped in a single transaction per call for speed
    and so a mid-batch failure doesn't leave the tables half-updated."""
    # Local imports keep this module importable without Django configured
    # (e.g. from a standalone script).
    from django.db import transaction
    from chat.models import FactProvenance, MetricFact

    counters = {"inserted": 0, "overwritten": 0, "duplicate": 0, "skipped": 0}
    if not normalized:
        return counters

    with transaction.atomic():
        for f in normalized:
            try:
                existing = MetricFact.objects.filter(
                    company=f["company"], period=f["period"],
                    metric_key=f["metric_key"],
                    statement_variant=f["statement_variant"],
                    unit=f["unit"],
                ).first()

                if existing is None:
                    new_row = MetricFact.objects.create(extracted_from=message, **f)
                    op = "insert"
                    counters["inserted"] += 1
                    _write_through_cache(new_row)
                elif existing.value == f["value"]:
                    op = "duplicate"
                    counters["duplicate"] += 1
                    # Refresh TTL so frequently-confirmed facts don't expire.
                    _write_through_cache(existing)
                else:
                    # Snapshot the prior value, then overwrite.
                    FactProvenance.objects.create(
                        company=existing.company, period=existing.period,
                        metric_key=existing.metric_key,
                        value=existing.value, unit=existing.unit,
                        statement_variant=existing.statement_variant,
                        source_doc=existing.source_doc,
                        source_page=existing.source_page,
                        extracted_from=existing.extracted_from,
                        confidence=existing.confidence,
                        operation="overwrite",
                    )
                    for k, v in f.items():
                        setattr(existing, k, v)
                    existing.extracted_from = message
                    existing.save()
                    op = "overwrite"
                    counters["overwritten"] += 1
                    _write_through_cache(existing)

                # Always log to provenance (even duplicates -- useful for
                # 'we saw the same number again from a different run').
                if op != "overwrite":
                    FactProvenance.objects.create(
                        extracted_from=message, operation=op, **f,
                    )
            except Exception as e:
                print(f"  [facts] persist row failed ({type(e).__name__}: {str(e)[:120]})")
                counters["skipped"] += 1

    return counters


def record_analysis_note(message, slots: dict, statement: str = "") -> None:
    """Mirror a non-trivial assistant answer into AnalysisNote for Slice 3
    recall. We skip notes with no companies/periods scope -- those don't
    benefit from recall (the bot can't match them against future questions).
    """
    from chat.models import AnalysisNote

    if not message:
        return
    companies = list((slots or {}).get("companies") or [])
    periods = list((slots or {}).get("periods") or [])
    if not companies and not periods:
        return
    try:
        AnalysisNote.objects.update_or_create(
            source_message=message,
            defaults={
                "scope": {
                    "companies": companies,
                    "periods": periods,
                    "statement": statement or "",
                    "metrics": list((slots or {}).get("metrics") or []),
                },
                "mode": message.mode or "",
                "body_md": message.content,
            },
        )
    except Exception as e:
        print(f"  [facts] analysis-note write failed ({type(e).__name__}: {str(e)[:120]})")


# ---------------------------------------------------------------------------
# Top-level entry point used by chat/views.py
# ---------------------------------------------------------------------------

def process_assistant_message(
    message,
    question: str,
    answer: str,
    sources: list[dict],
    slots: dict | None,
    statement: str = "",
) -> dict:
    """End-to-end: extract -> normalize -> persist -> record note.

    Returns counters from persist_facts (plus 'extracted' = total raw rows
    the LLM emitted). Never raises; logs and returns zeros on failure."""
    from nlu import known_companies

    counters = {"extracted": 0, "inserted": 0, "overwritten": 0,
                "duplicate": 0, "skipped": 0}
    try:
        whitelist = known_companies()
        allowed_docs = {(s.get("source") or "") for s in sources or []
                        if s.get("source")}
        raw = extract_facts_from_answer(question, answer, sources, whitelist)
        counters["extracted"] = len(raw)
        normalized = []
        for r in raw:
            n = normalize_fact(r, whitelist, allowed_docs)
            if n is not None:
                normalized.append(n)
            else:
                counters["skipped"] += 1
        write_counters = persist_facts(normalized, message=message)
        for k in ("inserted", "overwritten", "duplicate"):
            counters[k] = write_counters.get(k, 0)
        counters["skipped"] += write_counters.get("skipped", 0)
        record_analysis_note(message, slots, statement=statement)
    except Exception as e:
        print(f"  [facts] pipeline failed ({type(e).__name__}: {str(e)[:120]})")
    return counters
