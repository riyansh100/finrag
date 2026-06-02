"""Slice-3 proactive recall: find past AnalysisNote rows whose analytical
scope overlaps the current question, so the UI can surface
"you asked this 3 days ago — here's the prior answer + refresh".

The scope shape (set by facts.record_analysis_note) is:
    {"companies": [...], "periods": [...], "statement": "", "metrics": [...]}

We rank candidates by a weighted overlap of (company, period, statement),
filter under a confidence threshold, cap at N, then return JSON-safe dicts
ready to ship to the frontend.

Lookup is intentionally a single indexed SQL query (cheap, deterministic)
rather than vector similarity -- analysis scope is structured, and matching
on the structured fields gives sharper recall than embedding similarity for
this use case. We may add a semantic fallback later for free-text questions
that don't resolve to clean slots.
"""

from __future__ import annotations

from typing import Iterable


# Minimum score for a note to be considered "matching". Tuned conservatively
# so the user only sees a recall panel when it's clearly relevant. Below this
# the past note is too tangential to be useful and would just be noise.
MATCH_THRESHOLD = 0.5

# Hard cap on candidates returned -- the UI shows a panel, not a feed.
MAX_RECALLS = 3


def _to_set(xs) -> set[str]:
    return {str(x).strip() for x in (xs or []) if str(x).strip()}


def match_score(query_scope: dict, note_scope: dict) -> float:
    """Weighted overlap score in [0, 1].

    Weights chosen so:
      - identical scope -> 1.0
      - same company + same statement + ANY period overlap -> well over 0.5
      - same statement only -> below threshold (don't surface)
      - cross-company match -> hard-zero (different company is never a recall)

    Components:
      - company overlap: Jaccard. Hard gate -- if both sides name companies
        and they don't intersect, score is 0 (different company, not a recall).
      - period overlap: Jaccard.
      - statement match: 1.0 if exact, 0.5 if either side unspecified, 0 otherwise.
    """
    q_co = _to_set(query_scope.get("companies"))
    n_co = _to_set(note_scope.get("companies"))
    # Hard gate: distinct companies on both sides => not a recall candidate.
    if q_co and n_co and not (q_co & n_co):
        return 0.0
    co_overlap = (len(q_co & n_co) / len(q_co | n_co)
                  if (q_co or n_co) else 0.5)

    q_per = _to_set(query_scope.get("periods"))
    n_per = _to_set(note_scope.get("periods"))
    period_overlap = (len(q_per & n_per) / len(q_per | n_per)
                      if (q_per or n_per) else 0.5)

    q_stmt = (query_scope.get("statement") or "").lower().strip()
    n_stmt = (note_scope.get("statement") or "").lower().strip()
    if q_stmt and n_stmt:
        stmt_match = 1.0 if q_stmt == n_stmt else 0.0
    else:
        stmt_match = 0.5  # one side unspecified -> partial credit

    # Period overlap weighted heaviest -- two questions about Infosys are only
    # really "the same question" if they cover the same periods.
    return 0.5 * period_overlap + 0.3 * co_overlap + 0.2 * stmt_match


def _note_to_dict(note, score: float) -> dict:
    """JSON-safe shape for the API response / frontend panel."""
    return {
        "id":              note.id,
        "message_id":      note.source_message_id,
        "chat_id":         note.source_message.chat_id if note.source_message else None,
        "mode":            note.mode,
        "scope":           note.scope or {},
        "score":           round(score, 3),
        "created_at":      note.created_at.isoformat(),
        # Preview only -- the full body is fetched on demand when the user
        # clicks through to the prior chat. Keeps the response light.
        "preview":         (note.body_md or "")[:280],
        "body_length":     len(note.body_md or ""),
    }


def find_candidates(
    slots: dict,
    exclude_message_id: int | None = None,
    limit: int = MAX_RECALLS,
) -> list[dict]:
    """Return up to `limit` past AnalysisNotes matching the given slots.

    Strategy:
      1. Cheap SQL pre-filter: notes whose JSON scope mentions at least one
         of our companies OR one of our periods. SQLite's JSON1 makes this
         fast even without dedicated indexes once volumes grow.
      2. Python re-score using match_score() to handle the weighted overlap
         logic uniformly across DB backends.
      3. Filter under MATCH_THRESHOLD, sort by (score desc, recency desc).

    Returns [] when slots are empty (nothing to match against) or on any
    failure -- recall is a UX nicety, it must not break the answer flow."""
    if not slots:
        return []
    companies = list(_to_set(slots.get("companies")))
    periods = list(_to_set(slots.get("periods")))
    if not companies and not periods:
        # Nothing scoped to match against; bail rather than scan the whole
        # AnalysisNote table.
        return []

    try:
        from django.db.models import Q

        from chat.models import AnalysisNote

        # Pre-filter: contains-any on the JSON fields. icontains is good
        # enough -- companies are short slugs, periods are canonical labels,
        # both fit JSON-encoded substring matching.
        clauses = Q()
        for c in companies:
            clauses |= Q(scope__icontains=f'"{c}"')
        for p in periods:
            clauses |= Q(scope__icontains=f'"{p}"')

        qs = (AnalysisNote.objects
              .filter(clauses)
              .select_related("source_message")
              .order_by("-created_at"))
        if exclude_message_id is not None:
            qs = qs.exclude(source_message_id=exclude_message_id)

        # Re-score in Python. Cap the candidate list before scoring so a
        # giant history doesn't blow up the per-query cost.
        scored = []
        for note in qs[:50]:
            s = match_score(slots, note.scope or {})
            if s >= MATCH_THRESHOLD:
                scored.append((s, note))

        # Stable sort: score desc, then created_at desc (already from qs).
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_note_to_dict(n, s) for s, n in scored[:limit]]
    except Exception as e:
        print(f"  [recall] lookup failed ({type(e).__name__}: {str(e)[:120]})")
        return []
