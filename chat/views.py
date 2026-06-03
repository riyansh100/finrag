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
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

import facts as facts_pipeline
import nlu
import recall as analysis_recall
import uploads as upload_store

from . import rag
from .models import AnalysisNote, Chat, Message, UploadedDoc
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


def _upload_to_dict(u: UploadedDoc) -> dict:
    return {
        "id":           u.pk,
        "chat_id":      u.chat_id,
        "filename":     u.filename,
        "pages":        u.pages,
        "chunk_count":  u.chunk_count,
        "status":       u.status,
        "error":        u.error,
        "created_at":   u.created_at.isoformat(),
    }


@api_view(["GET", "POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def chat_uploads(request, chat_id):
    """GET  /api/chats/{id}/uploads        -> list of attached PDFs.
    POST /api/chats/{id}/uploads        -> multipart upload, indexes the PDF
        synchronously, returns the UploadedDoc row when ready.

    Same file re-uploaded into the same chat is a no-op: we return the
    existing row instead of re-indexing. PDFs > UPLOAD_MAX_MB are rejected
    with 400. Failures leave a row with status=failed + error message."""
    chat = get_object_or_404(Chat, pk=chat_id)

    if request.method == "GET":
        return Response([_upload_to_dict(u) for u in chat.uploads.all()])

    f = request.FILES.get("file")
    if f is None:
        return Response(
            {"detail": "Field 'file' (multipart) is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Size + MIME guard.
    max_bytes = config.UPLOAD_MAX_MB * 1024 * 1024
    if f.size > max_bytes:
        return Response(
            {"detail": f"File too large ({f.size} bytes); max is "
                       f"{config.UPLOAD_MAX_MB} MB."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    name = f.name or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        return Response(
            {"detail": "Only .pdf files are accepted."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    file_bytes = f.read()
    sha = upload_store.sha256_of(file_bytes)

    # Per-chat dedup: same bytes, same chat -> return the existing row.
    existing = chat.uploads.filter(sha256=sha).first()
    if existing is not None:
        return Response(_upload_to_dict(existing),
                        status=status.HTTP_200_OK)

    upload = UploadedDoc.objects.create(
        chat=chat, filename=name, sha256=sha,
        status=UploadedDoc.STATUS_PENDING,
    )
    try:
        counters = upload_store.ingest_pdf(upload, file_bytes)
        print(f"  [upload {upload.pk}] {name}: {counters}")
    except Exception as e:
        upload.status = UploadedDoc.STATUS_FAILED
        upload.error = f"ingest: {type(e).__name__}: {str(e)[:200]}"
        upload.save(update_fields=["status", "error", "updated_at"])

    upload.refresh_from_db()
    code = (status.HTTP_201_CREATED
            if upload.status == UploadedDoc.STATUS_READY
            else status.HTTP_400_BAD_REQUEST)
    return Response(_upload_to_dict(upload), status=code)


@api_view(["DELETE"])
def chat_upload_detail(request, chat_id, upload_id):
    """DELETE /api/chats/{id}/uploads/{upload_id}  -> drop Chroma collection,
    stored PDF, and the row. Idempotent on partially-indexed uploads."""
    chat = get_object_or_404(Chat, pk=chat_id)
    upload = get_object_or_404(UploadedDoc, pk=upload_id, chat=chat)
    upload_store.drop_upload(upload)
    upload.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


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

    # Optional list of UploadedDoc ids to search alongside the curated corpus.
    # Validated against THIS chat -- no cross-chat upload access.
    raw_upload_ids = body.get("upload_ids") or []
    if not isinstance(raw_upload_ids, list):
        return Response(
            {"detail": "'upload_ids' must be a list of integers."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    valid_upload_ids = list(
        chat.uploads.filter(
            pk__in=raw_upload_ids, status=UploadedDoc.STATUS_READY,
        ).values_list("pk", flat=True)
    )

    user_msg = Message.objects.create(chat=chat, role=Message.USER, content=question)

    result = rag.run_query(question, history=prior, mode=mode,
                           upload_ids=valid_upload_ids)

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
    # Slice-4: skip persistence when uploaded PDFs participated. Upload facts
    # don't belong in the canonical MetricFact cache (they're per-chat,
    # ephemeral relative to the curated corpus). This keeps cross-chat recall
    # honest.
    if valid_upload_ids:
        print(f"  [facts] msg#{assistant_msg.pk}: skipped "
              f"(upload-augmented answer)")
    try:
        if valid_upload_ids:
            slots = {}
            counters = {"extracted": 0}
        else:
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
