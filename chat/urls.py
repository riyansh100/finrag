"""URL routes for the chat app, mounted under /api/ by the project urlconf."""

from django.urls import path

from . import views

urlpatterns = [
    path("chats", views.chat_list, name="chat-list"),
    path("chats/<int:chat_id>", views.chat_detail, name="chat-detail"),
    path("chats/<int:chat_id>/messages", views.message_create, name="message-create"),
    path("modes", views.mode_list, name="mode-list"),
    # Slice-3: lookup past analyses matching a draft question. Used by the
    # composer to surface "you asked something like this before" before the
    # user even hits send.
    path("recall", views.recall_lookup, name="recall-lookup"),
    # Slice-3: full body of a single AnalysisNote (the preview shipped in
    # the recall list is truncated to 280 chars).
    path("notes/<int:note_id>", views.analysis_note_detail, name="note-detail"),
]
