"""API views for the chat app.

Endpoints (all under /api/):
  GET  /chats              list chats (sidebar)
  POST /chats              create a chat
  GET  /chats/{id}         one chat + its full message list
  DELETE /chats/{id}       delete a chat (cascades to messages)
  POST /chats/{id}/messages  ask a question -> runs the RAG pipeline
"""

import config
from django.shortcuts import get_object_or_404
from modes import DEFAULT_MODE, MODES, list_modes
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

import facts as facts_pipeline
import nlu
import recall as analysis_recall

from . import rag
from .models import AnalysisNote, Chat, Message
from .serializers import ChatDetailSerializer, ChatSerializer, MessageSerializer


@api_view(["GET"])
def recall_lookup(request):
    """GET /api/recall?question=...  ->  {"recall": [...]}.

    Pre-submit recall probe: the composer can call this as the user types
    (debounced) to surface a "you asked this before" panel without having to
    commit to a full RAG call. Uses the same NLU slot extractor as ask() so
    the matching is consistent."""
    question = (request.query_params.get("question") or "").strip()
    if not question:
        return Response({"recall": []})
    # No history -- the pre-submit path is meant to be cheap and stateless.
    slots = nlu.extract_slots(question, history=None) or {}
    # extract_slots returns the raw slot shape (companies, quarters, fys, ...);
    # we normalize to the (companies, periods, statement, metrics) shape the
    # recall matcher expects.
    from cache import _METRIC_KEYS_KNOWN  # noqa: F401 (just ensures lazy import)
    from nlu import slots_to_periods
    scope = {
        "companies": list(slots.get("companies") or []),
        "periods":   sorted(slots_to_periods(slots)) if slots else [],
        "statement": "",
        "metrics":   list(slots.get("metrics") or []),
    }
    hits = analysis_recall.find_candidates(scope)
    return Response({"recall": hits, "scope": scope})


@api_view(["GET"])
def analysis_note_detail(request, note_id):
    """GET /api/notes/{id}  ->  full body of one AnalysisNote.

    The recall list ships only a 280-char preview to keep responses small.
    Frontend hits this when the user expands a recall card."""
    note = get_object_or_404(AnalysisNote, pk=note_id)
    return Response({
        "id":          note.id,
        "message_id":  note.source_message_id,
        "chat_id":     note.source_message.chat_id if note.source_message else None,
        "mode":        note.mode,
        "scope":       note.scope or {},
        "body_md":     note.body_md,
        "created_at":  note.created_at.isoformat(),
    })


@api_view(["GET"])
def mode_list(request):
    """GET /api/modes -> [{id, label, description}, ...] plus the default.
    Frontend uses this to populate the mode dropdown."""
    return Response({"modes": list_modes(), "default": DEFAULT_MODE})


@api_view(["GET", "POST"])
def chat_list(request):
    """GET  /api/chats  -> list chats (sidebar).
    POST /api/chats  -> create a new chat; optional {"title": "..."}.
    """
    if request.method == "POST":
        title = (request.data or {}).get("title") or "New chat"
        chat = Chat.objects.create(title=title)
        return Response(ChatSerializer(chat).data, status=status.HTTP_201_CREATED)

    chats = Chat.objects.all()  # Meta.ordering -> most recently active first
    return Response(ChatSerializer(chats, many=True).data)


@api_view(["GET", "DELETE"])
def chat_detail(request, chat_id):
    """GET    /api/chats/{id}  -> one chat with its full message list.
    DELETE /api/chats/{id}  -> delete the chat (cascades to its messages).
    """
    chat = get_object_or_404(Chat, pk=chat_id)

    if request.method == "DELETE":
        chat.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    return Response(ChatDetailSerializer(chat).data)


@api_view(["POST"])
def message_create(request, chat_id):
    """POST /api/chats/{id}/messages  body: {"question": "..."}

    Saves the user turn, loads this chat's recent history from the DB, runs the
    RAG pipeline, saves the assistant turn (with sources + flags), and returns
    both messages.
    """
    chat = get_object_or_404(Chat, pk=chat_id)

    body = request.data or {}
    question = body.get("question", "").strip()
    if not question:
        return Response(
            {"detail": "Field 'question' is required and must be non-empty."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    mode = (body.get("mode") or DEFAULT_MODE).lower()
    if mode not in MODES:
        return Response(
            {"detail": f"Unknown mode '{mode}'. Valid: {sorted(MODES)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # History = prior turns only (before this question), trimmed to the last
    # HISTORY_TURNS messages. Read BEFORE saving the new user message.
    prior = list(
        chat.messages.order_by("-created_at")
        .values("role", "content")[: config.HISTORY_TURNS]
    )[::-1]  # back to chronological order

    user_msg = Message.objects.create(chat=chat, role=Message.USER, content=question)

    result = rag.run_query(question, history=prior, mode=mode)

    assistant_msg = Message.objects.create(
        chat=chat,
        role=Message.ASSISTANT,
        content=result["answer"],
        sources=result["sources"],
        flags=result["flags"],
        mode=result.get("mode") or mode,
    )

    # Slice-1 analytics layer: extract structured facts from this answer and
    # upsert into MetricFact (+ FactProvenance + AnalysisNote). Wrapped in a
    # broad try/except — the analytics pipeline MUST NEVER block or break the
    # user-facing answer flow.
    try:
        slots = result.get("slots") or {}
        counters = facts_pipeline.process_assistant_message(
            message=assistant_msg,
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            slots=slots,
            statement=slots.get("statement") or "",
        )
        if counters.get("extracted"):
            print(f"  [facts] msg#{assistant_msg.pk}: "
                  f"{counters['extracted']} extracted -> "
                  f"{counters['inserted']} new / "
                  f"{counters['overwritten']} updated / "
                  f"{counters['duplicate']} same / "
                  f"{counters['skipped']} skipped")
    except Exception as e:
        # Belt-and-suspenders: process_assistant_message already swallows,
        # but never let an analytics surprise reach the user.
        print(f"  [facts] view-level catch: {type(e).__name__}: {str(e)[:120]}")

    # Auto-title a brand-new chat from its first question; touch updated_at so it
    # rises to the top of the sidebar.
    if chat.title in ("", "New chat"):
        chat.title = question[:80]
    chat.save(update_fields=["title", "updated_at"])

    return Response(
        {
            "user_message": MessageSerializer(user_msg).data,
            "assistant_message": MessageSerializer(assistant_msg).data,
            "rewritten_query": result["rewritten_query"],
            # Slice-3: prior analyses with overlapping scope. List of
            # {id, message_id, chat_id, mode, scope, score, created_at,
            #  preview, body_length}. Empty when nothing matched. Frontend
            # renders a "Related past analysis" panel when non-empty.
            "recall": result.get("recall") or [],
        },
        status=status.HTTP_201_CREATED,
    )
