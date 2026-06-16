"""Dashboard data layer: structured financial time-series straight from the
MetricFact table. Pure Django ORM — no RAG, no LLM, no cloud calls. Powers both
the cross-document dashboard view and the prompt-driven charts in chat.

The shape returned is chart-ready: per company, a sorted period axis plus one
aligned value series per metric (None where a period is missing), so the
frontend can hand each series straight to Plotly.
"""

from __future__ import annotations

import re

# When the same metric exists in several units (revenue in both ₹crore and US$m),
# collapse to ONE series using this preference order. Ratios/EPS/DSO have their
# own natural units and simply fall through to whatever they were stored as.
_UNIT_PREFERENCE = ["inr_crore", "pct", "rupees", "days", "usd_million"]

# Human labels + the headline metrics the dashboard charts by default. The
# endpoint still returns every metric present; this just drives default charts
# and nice axis titles on the frontend.
METRIC_LABELS = {
    "revenue":              "Revenue",
    "operating_profit":     "Operating profit",
    "pat":                  "Net profit (PAT)",
    "total_assets":         "Total assets",
    "total_equity":         "Total equity",
    "total_liabilities":    "Total liabilities",
    "operating_margin_pct": "Operating margin %",
    "pat_margin_pct":       "Net margin %",
    "roe_pct":              "Return on equity %",
    "eps_basic":            "Basic EPS",
    "dso_days":             "DSO (days)",
    "trade_receivables":    "Trade receivables",
    "cash_and_equivalents": "Cash & equivalents",
}

# Default chart groupings for the dashboard. Each becomes one chart; metrics in
# a group share a y-axis (so they must be unit-compatible).
DEFAULT_CHARTS = [
    {"title": "Revenue", "metrics": ["revenue"], "kind": "bar"},
    {"title": "Profitability", "metrics": ["operating_profit", "pat"], "kind": "line"},
    {"title": "Margins %", "metrics": ["operating_margin_pct", "pat_margin_pct",
                                       "roe_pct"], "kind": "line"},
    {"title": "Balance sheet", "metrics": ["total_assets", "total_equity",
                                           "total_liabilities"], "kind": "line"},
]

_PERIOD_RE = re.compile(r"^(?:Q([1-4]))?FY(\d{2})$")

# Words that signal the user wants a visual, not (just) prose.
_CHART_INTENT_RE = re.compile(
    r"\b(chart|charts|plot|graph|graphs|trend|trends|visuali[sz]e|"
    r"bar chart|line chart|over time|time series)\b",
    re.IGNORECASE,
)


def period_sort_key(period: str):
    """Chronological sort key. 'Q3FY24' -> (24, 3); 'FY24' -> (24, 9) so an
    annual figure sorts after that year's quarters. Unparseable -> sorts last."""
    m = _PERIOD_RE.match(period or "")
    if not m:
        return (999, 9, period or "")
    q = int(m.group(1)) if m.group(1) else 9
    return (int(m.group(2)), q, "")


def _pick_unit(units: set[str]) -> str | None:
    for u in _UNIT_PREFERENCE:
        if u in units:
            return u
    return next(iter(sorted(units)), None)


def available_companies() -> list[str]:
    """Companies that have at least one fact, alphabetically."""
    from chat.models import MetricFact
    return sorted(MetricFact.objects.values_list("company", flat=True).distinct())


def company_series(company: str | None = None,
                   metrics: list[str] | None = None) -> dict:
    """Build chart-ready series from MetricFact.

    Returns:
        {
          "companies": [...],
          "data": {
            <company>: {
              "periods": [<sorted period labels>],
              "series": {
                <metric>: {"unit": <unit>, "label": <label>,
                           "values": [<aligned floats / None>]},
                ...
              }
            }, ...
          },
          "metric_labels": {...},
          "default_charts": [...],
        }
    """
    from chat.models import MetricFact

    qs = MetricFact.objects.all()
    if company:
        qs = qs.filter(company=company)
    if metrics:
        qs = qs.filter(metric_key__in=metrics)

    # nested: bucket[company][metric][unit][period] = float(value)
    bucket: dict = {}
    for r in qs.values("company", "period", "metric_key", "value", "unit"):
        c = r["company"]
        bucket.setdefault(c, {}).setdefault(r["metric_key"], {}) \
              .setdefault(r["unit"], {})[r["period"]] = float(r["value"])

    data: dict = {}
    for c, metric_map in bucket.items():
        # Period axis = union of every period this company has any fact for.
        periods = set()
        for unit_map in metric_map.values():
            for per_map in unit_map.values():
                periods.update(per_map.keys())
        # Keep ONE granularity. If the company has quarterly periods (QxFYyy),
        # drop the bare annual ones (FYyy) — they come from a single "year
        # ended" column in a quarterly report, so they're sparse and mixing
        # them into the quarterly axis just makes a confusing gap (the stray
        # "FY22" between Q3FY22 and Q1FY23).
        has_quarterly = any(_PERIOD_RE.match(p) and _PERIOD_RE.match(p).group(1)
                            for p in periods)
        if has_quarterly:
            periods = {p for p in periods
                       if _PERIOD_RE.match(p) and _PERIOD_RE.match(p).group(1)}
        period_axis = sorted(periods, key=period_sort_key)

        series = {}
        for metric, unit_map in metric_map.items():
            unit = _pick_unit(set(unit_map.keys()))
            per_map = unit_map.get(unit, {})
            series[metric] = {
                "unit": unit,
                "label": METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
                "values": [per_map.get(p) for p in period_axis],
            }
        data[c] = {"periods": period_axis, "series": series}

    return {
        "companies": sorted(data.keys()),
        "data": data,
        "metric_labels": METRIC_LABELS,
        "default_charts": DEFAULT_CHARTS,
    }


def has_chart_intent(question: str) -> bool:
    """True if the question explicitly asks for a chart/plot/trend."""
    return bool(_CHART_INTENT_RE.search(question or ""))


_PCT_METRICS = {"operating_margin_pct", "pat_margin_pct", "roe_pct"}


def _fmt_val(metric: str, v: float) -> str:
    if metric in _PCT_METRICS:
        return f"{v:.1f}%"
    if metric.endswith("_days"):
        return f"{v:.0f} days"
    if metric.startswith("eps"):
        return f"₹{v:.2f}"
    return f"₹{v:,.0f} cr"


def caption_for_chart(chart: dict) -> str:
    """Deterministic, LLM-free one-paragraph summary of a chart's data, so the
    text next to a chart is always consistent with it (the RAG/LLM answer reads
    PDF chunks and can disagree with the SQL series). First → last value with %
    change per metric, over the charted period span."""
    periods = chart.get("periods") or []
    lines = []
    for s in chart.get("series", []):
        vals = [(p, v) for p, v in zip(periods, s["values"]) if v is not None]
        if len(vals) < 2:
            continue
        (p0, v0), (p1, v1) = vals[0], vals[-1]
        metric = s["metric"]
        delta = ""
        if metric not in _PCT_METRICS and v0:
            pct = (v1 - v0) / abs(v0) * 100
            delta = f" ({'+' if pct >= 0 else ''}{pct:.0f}% over the period)"
        elif metric in _PCT_METRICS:
            delta = f" ({v1 - v0:+.1f} pts)"
        lines.append(f"- **{s['label']}**: {_fmt_val(metric, v0)} ({p0}) → "
                     f"{_fmt_val(metric, v1)} ({p1}){delta}")
    if not lines:
        return f"Charted **{chart['company'].upper()}** below."
    header = (f"**{chart['company'].upper()}** — {len(periods)} periods "
              f"({periods[0]}–{periods[-1]}). From the structured financials:")
    return header + "\n\n" + "\n".join(lines) + \
        "\n\n*Chart below is rendered directly from the metric store.*"


# Aliases tuned for casual CHART phrasing ("plot eps", "graph margins"), which
# is looser than the statement-extraction vocabulary in facts.CANONICAL_METRICS
# (that one has no bare "eps", "equity", "assets", or "margins"). Longest alias
# wins within a metric so "operating margin" beats "operating".
_CHART_ALIASES = {
    "revenue":              ["revenue", "topline", "sales"],
    "operating_profit":     ["operating profit", "ebit"],
    "pat":                  ["net profit", "profit after tax", "net income", "pat"],
    "total_assets":         ["total assets", "assets"],
    "total_equity":         ["total equity", "shareholders equity", "net worth", "equity"],
    "total_liabilities":    ["total liabilities", "liabilities"],
    "operating_margin_pct": ["operating margin"],
    "pat_margin_pct":       ["net margin", "pat margin", "profit margin"],
    "roe_pct":              ["return on equity", "roe"],
    "eps_basic":            ["basic eps", "earnings per share", "eps"],
    "dso_days":             ["dso", "days sales outstanding"],
    "trade_receivables":    ["trade receivables", "receivables"],
    "cash_and_equivalents": ["cash and cash equivalents", "cash"],
}
_MARGIN_GROUP = ["operating_margin_pct", "pat_margin_pct"]


def _metrics_from_text(question: str) -> list[str]:
    """Canonical metric keys named directly in the question text (word-boundary
    scan over _CHART_ALIASES). LLM-free, so chart prompts resolve the right
    metric without depending on the slot extractor."""
    q = (question or "").lower()
    found = []
    # Generic "margin(s)" with no specific qualifier -> both margin lines.
    if re.search(r"\bmargins?\b", q) and not re.search(
            r"operating margin|net margin|pat margin|profit margin", q):
        found.extend(_MARGIN_GROUP)
    for key, aliases in _CHART_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                if key not in found:
                    found.append(key)
                break
    return found


def chart_for_question(question: str, slots: dict | None) -> dict | None:
    """Build a chart spec for a prompt that asks to visualise something, or
    None if there's no chart intent / not enough data.

    LLM-free: chart intent is regex, the company/metrics come from the slots the
    RAG pipeline already resolved, and the values come straight from MetricFact.
    A metric needs >= 2 non-null points across the period axis to be worth a
    line; otherwise there's nothing to trend.

    Returns: {"company", "title", "kind", "periods",
              "series": [{"metric", "label", "unit", "values"}]}
    """
    if not has_chart_intent(question):
        return None
    slots = slots or {}

    companies = slots.get("companies") or []
    if len(companies) != 1:
        # Multi-company or unknown company -> the dashboard view is the better
        # surface; we only auto-chart an unambiguous single company here.
        return None
    company = companies[0]

    # Resolve requested metrics -> canonical keys. Priority:
    #   1. metrics the LLM slot extractor returned — now CONSTRAINED to the
    #      canonical key vocabulary and validated in nlu._validate, so it gets
    #      first crack on fuzzy phrasing without the old free-text noise;
    #   2. the deterministic text scan, as a fallback when the LLM is down /
    #      returned nothing (closed-vocabulary lookup never fails silently);
    #   3. revenue, as a last-resort default for a bare "chart/trend".
    import facts as facts_lib
    wanted = []
    for m in (slots.get("metrics") or []):
        key = facts_lib.canonicalize_metric(m)
        if key and key not in wanted:
            wanted.append(key)
    if not wanted:
        wanted = _metrics_from_text(question)
    if not wanted:
        wanted = ["revenue"]

    payload = company_series(company=company, metrics=wanted)
    cdata = payload["data"].get(company)
    if not cdata:
        return None

    series = []
    for metric in wanted:
        s = cdata["series"].get(metric)
        if not s:
            continue
        if sum(1 for v in s["values"] if v is not None) < 2:
            continue  # not enough points to trend
        series.append({"metric": metric, "label": s["label"],
                       "unit": s["unit"], "values": s["values"]})
    if not series:
        return None

    labels = ", ".join(s["label"] for s in series)
    return {
        "company": company,
        "title": f"{company.upper()} — {labels}",
        "kind": "bar" if len(series) == 1 and series[0]["metric"] == "revenue"
                else "line",
        "periods": cdata["periods"],
        "series": series,
    }
