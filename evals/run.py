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
        # show a 1-line preview
        first_line = (answer or "").strip().splitlines()[0] if answer else ""
        if first_line:
            print(f"    {DIM}↳ {first_line[:160]}{RESET}")

    print(f"    {DIM}({elapsed:.1f}s){RESET}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true",
                        help="Skip LLM; score retrieval + filter + numeric only")
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

    all_checks = []
    for i, case in enumerate(cases):
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
