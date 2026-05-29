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
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import rag
from .models import Chat, Message
from .serializers import ChatDetailSerializer, ChatSerializer, MessageSerializer


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

    question = (request.data or {}).get("question", "").strip()
    if not question:
        return Response(
            {"detail": "Field 'question' is required and must be non-empty."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # History = prior turns only (before this question), trimmed to the last
    # HISTORY_TURNS messages. Read BEFORE saving the new user message.
    prior = list(
        chat.messages.order_by("-created_at")
        .values("role", "content")[: config.HISTORY_TURNS]
    )[::-1]  # back to chronological order

    user_msg = Message.objects.create(chat=chat, role=Message.USER, content=question)

    result = rag.run_query(question, history=prior)

    assistant_msg = Message.objects.create(
        chat=chat,
        role=Message.ASSISTANT,
        content=result["answer"],
        sources=result["sources"],
        flags=result["flags"],
    )

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
        },
        status=status.HTTP_201_CREATED,
    )
