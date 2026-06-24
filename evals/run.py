"""FinRAG eval runner.

Usage:
    python evals/run.py                 # full set, with answer generation
    python evals/run.py --retrieval     # skip LLM (faster); score retrieval + filter only
    python evals/run.py --case 3        # run a single case index

Scoring per case:
    retrieval     PASS if every expect_source filename appears in retrieved chunks.
    filter        PASS if detected source_filter == expect_filter.
    numeric       PASS if is_numeric_question() == expect_numeric.
    answer        PASS if every expect_substring appears in the answer (case-insensitive)
                  AND no forbid_substring appears. Skipped when --retrieval.

Exit code: 0 if all scored checks pass, else 1.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

# Make parent dir importable when running as `python evals/run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Initialise Django so the fact-cache and recall layers (which hit the ORM) run
# during evals instead of silently no-op'ing with AppRegistryNotReady. Mirrors
# what the server does, so cache short-circuit behaviour is exercised too.
import os  # noqa: E402
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finrag_backend.settings")
import django  # noqa: E402
django.setup()

from query import ask  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _check_retrieval(docs, expected):
    expected = _as_list(expected)
    if not expected:
        return True, "no source expected"
    sources = {d.metadata.get("source") for d in docs}
    missing = [e for e in expected if e not in sources]
    if missing:
        return False, f"missing: {missing}"
    return True, f"all {len(expected)} expected source(s) present"


def _check_substrings(answer, required, forbidden):
    answer_l = (answer or "").lower()
    missing = [s for s in _as_list(required) if s.lower() not in answer_l]
    bad = [s for s in _as_list(forbidden) if s.lower() in answer_l]
    if missing or bad:
        notes = []
        if missing:
            notes.append(f"missing: {missing}")
        if bad:
            notes.append(f"forbidden present: {bad}")
        return False, "; ".join(notes)
    return True, "all required substrings present"


def _fmt(passed, label, detail=""):
    marker = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    line = f"    {marker} {label}"
    if detail:
        line += f"  {DIM}{detail}{RESET}"
    return line


def run_case(idx, case, retrieval_only=False):
    q = case["question"]
    print(f"\n[{idx}] {YELLOW}{q}{RESET}")

    t0 = time.time()
    # ask() does the full slot/atom resolution; we always call it so the
    # company/period/atom checks are scored the same way in both modes. Under
    # --retrieval we just skip reading the (LLM-generated) answer.
    result = ask(q, skip_generation=retrieval_only)
    docs = result["sources"]
    answer = None if retrieval_only else result["answer"]
    detected_filter = result["filtered_to"]
    numeric = result["numeric"]
    company = result.get("company_filter")
    period = result.get("period_filter")
    atoms_fired = bool(result.get("atoms"))
    verification = result.get("verification")
    elapsed = time.time() - t0

    checks = []

    ok, detail = _check_retrieval(docs, case.get("expect_source"))
    checks.append(("retrieval", ok))
    print(_fmt(ok, "retrieval", detail))

    exp_filter = case.get("expect_filter")
    ok = detected_filter == exp_filter
    checks.append(("filter", ok))
    print(_fmt(ok, "filter", f"got={detected_filter!r} exp={exp_filter!r}"))

    if "expect_company" in case:
        exp = case["expect_company"]
        ok = company == exp
        checks.append(("company", ok))
        print(_fmt(ok, "company", f"got={company!r} exp={exp!r}"))

    if "expect_period" in case:
        exp = case["expect_period"]
        ok = period == exp
        checks.append(("period", ok))
        print(_fmt(ok, "period", f"got={period!r} exp={exp!r}"))

    if "expect_atoms" in case:
        exp = bool(case["expect_atoms"])
        ok = atoms_fired == exp
        checks.append(("atoms", ok))
        n = len(result.get("atoms") or [])
        print(_fmt(ok, "atoms", f"got={atoms_fired}({n}) exp={exp}"))

    exp_numeric = case.get("expect_numeric")
    if exp_numeric is not None:
        ok = numeric == exp_numeric
        checks.append(("numeric", ok))
        print(_fmt(ok, "numeric-intent", f"got={numeric} exp={exp_numeric}"))

    if not retrieval_only:
        ok, detail = _check_substrings(
            answer,
            case.get("expect_substrings"),
            case.get("forbid_substrings"),
        )
        checks.append(("answer", ok))
        print(_fmt(ok, "answer", detail))

        if "expect_verify" in case:
            exp = case["expect_verify"]
            got = (verification or {}).get("status", "n/a")
            ok = got == exp
            checks.append(("verify", ok))
            extra = ""
            if verification and verification.get("unverified"):
                extra = f" unverified={verification['unverified']}"
            print(_fmt(ok, "verify", f"got={got!r} exp={exp!r}{extra}"))

        # show a 1-line preview
        first_line = (answer or "").strip().splitlines()[0] if answer else ""
        if first_line:
            print(f"    {DIM}↳ {first_line[:160]}{RESET}")

    print(f"    {DIM}({elapsed:.1f}s){RESET}")
    return checks


def run_case_nlu(idx, case):
    """Score ONLY the deterministic query-understanding checks.

    Calls the standalone detect_* parsers directly instead of ask(), so this
    needs no Chroma index and no Ollama — it runs in CI on a bare runner.
    Covers the company / period / filter / numeric-intent expectations from
    qa.yaml; retrieval and answer checks are out of scope here (they need the
    vectorstore + LLM).
    """
    import query  # parsing layer; already pulled in via `from query import ask`

    q = case["question"]
    print(f"\n[{idx}] {YELLOW}{q}{RESET}")
    checks = []

    if "expect_company" in case:
        got, exp = query.detect_company_filter(q), case["expect_company"]
        ok = got == exp
        checks.append(("company", ok))
        print(_fmt(ok, "company", f"got={got!r} exp={exp!r}"))

    if "expect_period" in case:
        got, exp = query.detect_period_filter(q), case["expect_period"]
        # Mirror the pipeline: ask() drops a detected period whose FY is outside
        # the corpus range, but the raw parser doesn't. Apply the same guard so
        # this check matches result["period_filter"] (e.g. "Q4 FY27" -> None).
        if got:
            import re as _re
            import nlu
            m = _re.search(r"FY(\d{2})$", got)
            if m and int(m.group(1)) not in nlu.known_fys():
                got = None
        ok = got == exp
        checks.append(("period", ok))
        print(_fmt(ok, "period", f"got={got!r} exp={exp!r}"))

    if "expect_filter" in case:
        got, exp = query.detect_source_filter(q), case["expect_filter"]
        ok = got == exp
        checks.append(("filter", ok))
        print(_fmt(ok, "filter", f"got={got!r} exp={exp!r}"))

    exp_numeric = case.get("expect_numeric")
    if exp_numeric is not None:
        got = query.is_numeric_question(q)
        ok = got == exp_numeric
        checks.append(("numeric", ok))
        print(_fmt(ok, "numeric-intent", f"got={got} exp={exp_numeric}"))

    return checks


def run_faithfulness(cases, idx_offset):
    """Aggregate answer-grounding across the eval set into one trust metric.

    For each question we generate a real answer and reuse verify.py's tracer
    (already attached as result["verification"]): it checks every comma-grouped
    figure in the answer against the retrieved chunks + MetricFact store. A
    figure found nowhere is a hallucination.

        groundedness = traced figures / checked figures
        hallucination rate = 1 - groundedness

    Needs the LLM + vectorstore + Ollama (real generation), so this is a local
    metric, not part of CI. Exits non-zero if anything went unverified, so it
    can double as a gate later.
    """
    total_checked = total_traced = 0
    flagged = []

    for i, case in enumerate(cases):
        q = case["question"]
        result = ask(q)
        v = result.get("verification") or {}
        checked, traced = v.get("checked", 0), v.get("traced", 0)
        unverified = v.get("unverified", [])
        total_checked += checked
        total_traced += traced

        status = v.get("status", "n/a")
        color = GREEN if not unverified else RED
        print(f"[{i + idx_offset}] {color}{status:<7}{RESET} "
              f"traced {traced}/{checked}  {YELLOW}{q[:55]}{RESET}")
        if unverified:
            print(f"    {RED}unverified figures: {unverified}{RESET}")
            flagged.append((q, unverified))

    print()
    print("=" * 60)
    if total_checked == 0:
        print("  no comma-grouped figures to check across the set")
        sys.exit(0)
    groundedness = total_traced / total_checked
    print(f"  figures checked     {total_checked}")
    print(f"  figures grounded    {total_traced}")
    print(f"  {'GROUNDEDNESS':<18} {GREEN if not flagged else YELLOW}"
          f"{groundedness:.0%}{RESET}")
    print(f"  {'HALLUCINATION':<18} {RED if flagged else GREEN}"
          f"{1 - groundedness:.0%}{RESET}  ({len(flagged)} answer(s) flagged)")
    print("=" * 60)
    sys.exit(0 if not flagged else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true",
                        help="Skip LLM; score retrieval + filter + numeric only")
    parser.add_argument("--nlu", action="store_true",
                        help="Deterministic query-understanding checks only "
                             "(no vectorstore, no Ollama) — the CI-safe subset")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Generate answers and report groundedness / "
                             "hallucination rate via verify.py (needs LLM)")
    parser.add_argument("--case", type=int, default=None,
                        help="Run only this case index (0-based)")
    parser.add_argument("--file", default=str(Path(__file__).with_name("qa.yaml")))
    args = parser.parse_args()

    with open(args.file) as f:
        spec = yaml.safe_load(f)
    cases = spec["cases"]

    if args.case is not None:
        cases = [cases[args.case]]
        idx_offset = args.case
    else:
        idx_offset = 0

    if args.faithfulness:
        run_faithfulness(cases, idx_offset)
        return

    all_checks = []
    for i, case in enumerate(cases):
        if args.nlu:
            checks = run_case_nlu(i + idx_offset, case)
        else:
            checks = run_case(i + idx_offset, case, retrieval_only=args.retrieval)
        all_checks.extend(checks)

    print()
    print("=" * 60)
    by_kind = {}
    for kind, ok in all_checks:
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += int(ok)
        by_kind[kind][1] += 1
    for kind, (p, t) in sorted(by_kind.items()):
        color = GREEN if p == t else RED
        print(f"  {kind:<10} {color}{p}/{t}{RESET}")
    total_pass = sum(int(ok) for _, ok in all_checks)
    total = len(all_checks)
    overall = GREEN if total_pass == total else RED
    print(f"  {'OVERALL':<10} {overall}{total_pass}/{total}{RESET}")
    print("=" * 60)

    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
