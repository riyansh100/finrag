"""Domain pack loader.

A *domain pack* is a YAML file (packs/<name>/pack.yaml) holding everything the
generic FinRAG engine needs to know about a specific corpus: the entity
registry, fiscal calendar, metric vocabulary, units, currency detection, and
prompt fragments. Swapping `config.DOMAIN_PACK` retargets the engine at a
different domain with no code changes.

All domain-specific lookups in the engine go through the singleton returned by
`get_pack()`. The pack is parsed once and cached; call sites import the small
helper functions at the bottom of this module rather than reaching into the raw
dict.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

import config


# ---------------------------------------------------------------------------
# Parser specs: filename -> canonical period metadata
# ---------------------------------------------------------------------------

def _fy_from_group(value: str, mode: str) -> int:
    """Turn a regex-captured year string into a two-digit fiscal year.

    full_year -> the year IS the FY (Infosys q1-2026.pdf -> FY26).
    end_year  -> END year of an annual range ("2024-25"/"2024-2025" -> FY25).
    """
    n = int(value)
    if mode == "full_year":
        return n % 100
    if mode == "end_year":
        return n if n < 100 else n % 100
    raise ValueError(f"unknown fy mode: {mode!r}")


class _ParserSpec:
    """One compiled filename->period rule for a company."""

    def __init__(self, slug: str, spec: dict, fiscal: "_Fiscal"):
        self.slug = slug
        self.doc_type = spec["doc_type"]
        self.quarter_group = spec.get("quarter_group")
        fy = spec["fy"]
        self.fy_group = fy["group"]
        self.fy_mode = fy["mode"]
        self._fiscal = fiscal
        self._re = re.compile(spec["pattern"], re.IGNORECASE)

    def match(self, filename: str, with_company: bool = True) -> dict | None:
        m = self._re.match(filename or "")
        if not m:
            return None
        quarter = int(m.group(self.quarter_group)) if self.quarter_group else None
        fy = _fy_from_group(m.group(self.fy_group), self.fy_mode)
        meta = {
            "doc_type": self.doc_type,
            "period": self._fiscal.period_label(quarter, fy),
            "quarter": quarter,
            "fy": fy,
        }
        if with_company:
            meta["company"] = self.slug
        return meta


# ---------------------------------------------------------------------------
# Fiscal calendar
# ---------------------------------------------------------------------------

class _Fiscal:
    def __init__(self, cfg: dict):
        self.start_month = cfg.get("start_month", 4)
        self._fy_label = cfg.get("fy_label", "FY{fy:02d}")
        self._quarter_label = cfg.get("quarter_label", "Q{quarter}FY{fy:02d}")
        self.default_latest_fy = cfg.get("default_latest_fy", 26)

    def period_label(self, quarter: int | None, fy: int) -> str:
        if quarter:
            return self._quarter_label.format(quarter=quarter, fy=fy)
        return self._fy_label.format(fy=fy)


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

class Pack:
    def __init__(self, data: dict):
        self.name = data.get("name", "unnamed")
        self.description = data.get("description", "")
        self._raw = data

        self.fiscal = _Fiscal(data.get("fiscal") or {})

        # Entity registry ----------------------------------------------------
        self._entities = data.get("entities") or []
        self._slugs: list[str] = [e["slug"] for e in self._entities]
        # slug -> list of compiled parser specs
        self._parsers: dict[str, list[_ParserSpec]] = {}
        # token alias -> slug ; (phrase, slug) for multi-word aliases
        self._token_aliases: dict[str, str] = {}
        self._phrase_aliases: list[tuple[str, str]] = []
        for e in self._entities:
            slug = e["slug"]
            self._parsers[slug] = [
                _ParserSpec(slug, p, self.fiscal) for p in (e.get("parsers") or [])
            ]
            for alias in e.get("aliases") or []:
                a = alias.strip().lower()
                if " " in a:
                    self._phrase_aliases.append((a, slug))
                else:
                    self._token_aliases[a] = slug
        # Longer phrases first so the most specific alias wins.
        self._phrase_aliases.sort(key=lambda t: -len(t[0]))

        # Metrics / units ----------------------------------------------------
        self.metrics: dict[str, list[str]] = data.get("metrics") or {}
        self._metric_lookup = self._build_metric_lookup()
        self.metric_tail_noise: list[str] = data.get("metric_tail_noise") or []
        self._unit_lookup = self._build_unit_lookup(data.get("units") or {})

        # Currency detection -------------------------------------------------
        self.currency_detection = data.get("currency_detection") or {}

        # Prompts ------------------------------------------------------------
        self.prompts: dict[str, str] = data.get("prompts") or {}

    # -- entities -----------------------------------------------------------

    def company_slugs(self) -> list[str]:
        return list(self._slugs)

    def parsers_for(self, slug: str) -> list[_ParserSpec]:
        return self._parsers.get(slug.lower(), [])

    def parse_filename(self, company_folder: str, filename: str) -> dict | None:
        """data/<company>/<file> -> period metadata, or a permissive stub for an
        unrecognised filename inside a known company folder, else None."""
        slug = company_folder.lower()
        for spec in self.parsers_for(slug):
            meta = spec.match(filename, with_company=True)
            if meta is not None:
                return meta
        if slug in self._parsers:
            return {"company": slug, "doc_type": "unknown",
                    "period": None, "quarter": None, "fy": None}
        return None

    def detect_upload_meta(self, filename: str) -> dict:
        """Best-effort period for an ad-hoc upload (no company folder). Tries
        every company's parsers and returns the first period match WITHOUT a
        company tag. Empty dict if nothing matches. Never raises."""
        name = (filename or "").strip()
        for slug in self._slugs:
            for spec in self.parsers_for(slug):
                meta = spec.match(name, with_company=False)
                if meta is not None:
                    return meta
        return {}

    def token_aliases(self) -> dict[str, str]:
        return dict(self._token_aliases)

    def phrase_aliases(self) -> list[tuple[str, str]]:
        return list(self._phrase_aliases)

    def alias_hint(self) -> str:
        """One-line human hint mapping aliases to slugs, for LLM prompts.
        E.g. '"infy"/"infosys" -> "infosys". "reliance"/"riil"/... -> "riil".'"""
        parts = []
        for e in self._entities:
            aliases = e.get("aliases") or [e["slug"]]
            quoted = "/".join(f'"{a}"' for a in aliases)
            parts.append(f'{quoted} -> "{e["slug"]}".')
        return " ".join(parts)

    def alias_regex(self) -> str:
        """Regex alternation of every company alias (and slug), longest-first,
        for word-boundary entity-hint matching. Already regex-escaped."""
        terms = set(self._slugs) | set(self._token_aliases) \
            | {p for p, _ in self._phrase_aliases}
        ordered = sorted(terms, key=len, reverse=True)
        return "|".join(re.escape(t) for t in ordered)

    def detect_companies(self, *questions: str) -> set[str]:
        """Set of company slugs mentioned across the given question strings."""
        matched: set[str] = set()
        for q in questions:
            if not q:
                continue
            ql = q.lower()
            for phrase, slug in self._phrase_aliases:
                if phrase in ql:
                    matched.add(slug)
            for tok in re.findall(r"[a-z]+", ql):
                if tok in self._token_aliases:
                    matched.add(self._token_aliases[tok])
        return matched

    # -- metrics / units ----------------------------------------------------

    def _build_metric_lookup(self) -> dict[str, str]:
        rev: dict[str, str] = {}
        for canonical, aliases in self.metrics.items():
            rev[canonical.lower()] = canonical
            for alias in aliases:
                rev[alias.lower()] = canonical
        return rev

    def _build_unit_lookup(self, units: dict) -> dict[str, str]:
        rev: dict[str, str] = {}
        for canonical, aliases in units.items():
            rev[canonical.lower()] = canonical
            for alias in aliases:
                rev[str(alias).lower()] = canonical
        return rev

    def metric_lookup(self) -> dict[str, str]:
        return self._metric_lookup

    def unit_lookup(self) -> dict[str, str]:
        return self._unit_lookup

    def canonical_metrics(self) -> dict[str, list[str]]:
        return self.metrics


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

PACKS_DIR = config.BASE_DIR / "packs"


@lru_cache(maxsize=None)
def get_pack(name: str | None = None) -> Pack:
    """Load and cache the active domain pack. `name` defaults to
    config.DOMAIN_PACK."""
    pack_name = name or getattr(config, "DOMAIN_PACK", "finance-india")
    path = PACKS_DIR / pack_name / "pack.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"domain pack {pack_name!r} not found at {path}. "
            f"Set config.DOMAIN_PACK to a folder under {PACKS_DIR}/."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Pack(data)
