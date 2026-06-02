"""Slice-2 fact cache: Redis L1 in front of MetricFact (SQLite).

Three roles:
  1. LOOKUP    -- given (companies, periods, [metrics|statement]), return all
                  matching facts. Redis is checked first; misses fall through
                  to SQLite and the SQLite result is back-filled into Redis.
  2. INVALIDATE-- on every persist/overwrite, write-through to Redis so the
                  next read is consistent.
  3. INJECT    -- format cached facts into a synthetic LangChain Document the
                  retriever can splice into the LLM context.

Redis is OPTIONAL. If unreachable (no server, wrong URL, network blip) the
module silently degrades to SQLite-only. The user-facing path NEVER fails
because of cache trouble -- the worst case is a slower lookup.

Key shape:
  fact:<company>:<period>:<metric_key>:<statement_variant>:<unit>

Value: JSON of the MetricFact row's relevant fields. statement_variant of ""
(unspecified) is encoded as "_" in the key so it's still a non-empty token.
"""

from __future__ import annotations

import json
from decimal import Decimal
from itertools import product
from typing import Iterable

from langchain_core.documents import Document

import config


# ---------------------------------------------------------------------------
# Redis connection (lazy, fault-tolerant)
# ---------------------------------------------------------------------------

_REDIS = None
_REDIS_DISABLED = False  # latched True after the first connection failure


def _get_redis():
    """Return a connected redis client or None. Latches OFF after one failure
    so we don't pay reconnect cost on every lookup when the server is down."""
    global _REDIS, _REDIS_DISABLED
    if not config.FACT_CACHE_ENABLED or not config.REDIS_URL:
        return None
    if _REDIS_DISABLED:
        return None
    if _REDIS is not None:
        return _REDIS
    try:
        import redis  # local import: optional dep, install separately
        client = redis.from_url(config.REDIS_URL, decode_responses=True,
                                socket_connect_timeout=0.5,
                                socket_timeout=0.5)
        client.ping()
        _REDIS = client
        return client
    except Exception as e:
        print(f"  [cache] Redis disabled ({type(e).__name__}: {str(e)[:80]})")
        _REDIS_DISABLED = True
        return None


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _key(company, period, metric, variant, unit):
    """Stable key. statement_variant of '' becomes '_' so the segment is
    never empty (Redis tolerates it, but it makes SCAN patterns clearer)."""
    return f"fact:{company}:{period}:{metric}:{variant or '_'}:{unit}"


def _fact_to_dict(f) -> dict:
    """MetricFact row -> JSON-safe dict for Redis storage / context injection."""
    return {
        "company":           f.company,
        "period":            f.period,
        "metric_key":        f.metric_key,
        "value":             str(f.value),  # Decimal -> str (preserves precision)
        "unit":              f.unit,
        "statement_variant": f.statement_variant or "",
        "source_doc":        f.source_doc or "",
        "source_page":       f.source_page,
        "confidence":        f.confidence,
    }


# ---------------------------------------------------------------------------
# Write-through (called from facts.persist_facts after every commit)
# ---------------------------------------------------------------------------

def write_through(fact_row) -> None:
    """Push one MetricFact row into Redis. No-op if Redis is unavailable."""
    r = _get_redis()
    if r is None:
        return
    try:
        k = _key(fact_row.company, fact_row.period, fact_row.metric_key,
                 fact_row.statement_variant, fact_row.unit)
        r.set(k, json.dumps(_fact_to_dict(fact_row)),
              ex=config.FACT_CACHE_TTL_SEC)
    except Exception as e:
        print(f"  [cache] write-through failed ({type(e).__name__}: {str(e)[:80]})")


def invalidate(company, period, metric, variant="", unit=None) -> None:
    """Drop a specific key (or scan-and-drop all units when unit=None).
    Called from re-ingest paths; not used by the answer flow."""
    r = _get_redis()
    if r is None:
        return
    try:
        if unit:
            r.delete(_key(company, period, metric, variant, unit))
        else:
            pattern = _key(company, period, metric, variant, "*")
            for k in r.scan_iter(match=pattern, count=200):
                r.delete(k)
    except Exception as e:
        print(f"  [cache] invalidate failed ({type(e).__name__}: {str(e)[:80]})")


# ---------------------------------------------------------------------------
# Statement -> default metric set. Used when the user asks for a whole
# statement ("balance sheet") without naming specific line items, so we know
# which cells the cache needs to cover for full short-circuit.
# ---------------------------------------------------------------------------

def _METRIC_KEYS_KNOWN():
    """Set of canonical metric keys. Wrapped in a function so we don't import
    facts.py at module load (avoids a circular import on cold start)."""
    from facts import CANONICAL_METRICS
    return set(CANONICAL_METRICS.keys())


STATEMENT_METRICS = {
    "balance sheet": [
        "total_assets", "total_equity", "total_liabilities",
        "cash_and_equivalents", "current_investments",
        "non_current_investments", "trade_receivables", "ppe",
    ],
    "profit and loss": [
        "revenue", "cost_of_sales", "gross_profit",
        "total_operating_exp", "operating_profit", "operating_margin_pct",
        "other_income", "pbt", "tax", "pat", "pat_margin_pct",
        "eps_basic", "eps_diluted",
    ],
    "cash flow": [
        "cash_from_operations", "cash_from_investing", "cash_from_financing",
    ],
}


def expected_metrics_for(slots: dict) -> list[str]:
    """The metric set the cache must cover for full short-circuit.
    Priority: explicit metrics in slots -> statement-default metrics -> []."""
    from facts import canonicalize_metric  # local: avoid cycle

    explicit = []
    for m in (slots or {}).get("metrics") or []:
        c = canonicalize_metric(m)
        if c and c not in explicit:
            explicit.append(c)
    if explicit:
        return explicit
    statement = ((slots or {}).get("statement") or "").lower()
    return list(STATEMENT_METRICS.get(statement, []))


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _lookup_one_redis(company, period, metric) -> list[dict]:
    """Redis-side lookup for one (company, period, metric). Returns all rows
    across variants/units (rare to have >1, but possible: standalone +
    consolidated, INR + USD)."""
    r = _get_redis()
    if r is None:
        return []
    try:
        pattern = f"fact:{company}:{period}:{metric}:*:*"
        out = []
        for k in r.scan_iter(match=pattern, count=200):
            raw = r.get(k)
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out
    except Exception as e:
        print(f"  [cache] redis lookup failed ({type(e).__name__}: {str(e)[:80]})")
        return []


def _lookup_one_sqlite(company, period, metric) -> list[dict]:
    """SQLite fallback. Also back-fills Redis with whatever it finds."""
    from chat.models import MetricFact

    rows = list(MetricFact.objects.filter(
        company=company, period=period, metric_key=metric,
    ))
    out = []
    for row in rows:
        d = _fact_to_dict(row)
        out.append(d)
        write_through(row)  # warm the cache for next time
    return out


def lookup_facts(
    companies: Iterable[str],
    periods: Iterable[str],
    metrics: Iterable[str] | None = None,
    statement: str = "",
) -> list[dict]:
    """Resolve (company, period, metric) -> list of fact dicts. Redis first,
    SQLite fallback per cell. When `metrics` is empty AND `statement` is set,
    the statement's default metric list is used. When both are empty, ALL
    facts for the given (company, period) cross-product are returned (SQLite
    bulk query -- skips the per-cell loop)."""
    companies = list(companies or [])
    periods = list(periods or [])
    if not companies or not periods:
        return []

    # Canonicalize metric names so callers can pass anything in -- NLU emits
    # free text ("operating profit"), but MetricFact stores canonical keys
    # ("operating_profit"). Drop unknown names rather than mis-keying.
    from facts import canonicalize_metric

    metric_list = []
    for m in (metrics or []):
        c = canonicalize_metric(m) or (m if m in _METRIC_KEYS_KNOWN() else None)
        if c and c not in metric_list:
            metric_list.append(c)
    if not metric_list and statement:
        metric_list = STATEMENT_METRICS.get(statement.lower(), [])

    # Open-ended ask ("anything you have for infosys Q3FY24") -> bulk SQLite.
    if not metric_list:
        from chat.models import MetricFact
        rows = MetricFact.objects.filter(
            company__in=companies, period__in=periods,
        )
        results = []
        for row in rows:
            results.append(_fact_to_dict(row))
            write_through(row)
        return results

    # Targeted ask -> per-cell, Redis-first.
    results = []
    for company, period, metric in product(companies, periods, metric_list):
        hits = _lookup_one_redis(company, period, metric)
        if not hits:
            hits = _lookup_one_sqlite(company, period, metric)
        results.extend(hits)
    return results


def lookup_for_slots(slots: dict) -> list[dict]:
    """Convenience wrapper that pulls (companies, periods, metrics, statement)
    out of a slot dict and calls lookup_facts."""
    if not slots:
        return []
    return lookup_facts(
        companies=slots.get("companies") or [],
        periods=slots.get("periods") or [],
        metrics=slots.get("metrics") or [],
        statement=slots.get("statement") or "",
    )


# ---------------------------------------------------------------------------
# Coverage check + context injection
# ---------------------------------------------------------------------------

def coverage(facts: list[dict], slots: dict) -> tuple[int, int]:
    """How many of the requested (company, period, metric) cells did the cache
    satisfy? Returns (hits, total). total=0 means 'we can't define coverage'
    (e.g. user asked an open-ended question)."""
    companies = (slots or {}).get("companies") or []
    periods = (slots or {}).get("periods") or []
    metrics = expected_metrics_for(slots)
    if not (companies and periods and metrics):
        return (0, 0)
    have = {(f["company"], f["period"], f["metric_key"]) for f in facts}
    want = set(product(companies, periods, metrics))
    return (len(want & have), len(want))


def is_full_coverage(facts: list[dict], slots: dict) -> bool:
    """True iff the cache covers EVERY requested cell. When True, the caller
    may short-circuit RAG (controlled by FACT_CACHE_SHORTCIRCUIT_RAG)."""
    hits, total = coverage(facts, slots)
    return total > 0 and hits == total


def format_facts_as_context_chunk(facts: list[dict]) -> Document | None:
    """Render cached facts as a single synthetic LangChain Document that the
    LLM treats as authoritative context. Citations preserved so the answer
    can still cite [filename p.N]. Returns None when there are no facts."""
    if not facts:
        return None

    # Group by (company, period) for readability.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for f in facts:
        grouped.setdefault((f["company"], f["period"]), []).append(f)

    lines = [
        "CACHED ANALYST FACTS — these values were previously extracted from "
        "the source documents and validated. Treat them as AUTHORITATIVE and "
        "use them verbatim. Each row includes its original source citation.",
        "",
    ]
    for (company, period), rows in sorted(grouped.items()):
        lines.append(f"## {company.upper()} · {period}")
        rows.sort(key=lambda r: r["metric_key"])
        for r in rows:
            variant = f" ({r['statement_variant']})" if r["statement_variant"] else ""
            cite = ""
            if r.get("source_doc"):
                page = r.get("source_page")
                cite = f" [{r['source_doc']}" + (f" p.{page}" if page else "") + "]"
            lines.append(
                f"- {r['metric_key']}{variant}: {r['value']} {r['unit']}{cite}"
            )
        lines.append("")

    body = "\n".join(lines).rstrip()
    return Document(
        page_content=body,
        metadata={
            "source": "<cached-facts>",
            "page": 0,
            "type": "cached_facts",
            "synthetic": True,
        },
    )
