"""URL routes for the chat app, mounted under /api/ by the project urlconf."""

from django.urls import path

from . import views

urlpatterns = [
    path("chats", views.chat_list, name="chat-list"),
    path("chats/<int:chat_id>", views.chat_detail, name="chat-detail"),
    path("chats/<int:chat_id>/messages", views.message_create, name="message-create"),
    # Slice-4: per-chat on-the-fly PDF uploads. Indexed into their own Chroma
    # collection and searched alongside the corpus when attached to a turn.
    path("chats/<int:chat_id>/uploads",
         views.chat_uploads, name="chat-uploads"),
    path("chats/<int:chat_id>/uploads/<int:upload_id>",
         views.chat_upload_detail, name="chat-upload-detail"),
    path("modes", views.mode_list, name="mode-list"),
    # Cross-document financial dashboard: chart-ready series straight from
    # MetricFact (SQL only, no RAG). Optional ?company=<slug>.
    path("dashboard", views.dashboard_data, name="dashboard-data"),
    # Slice-3: lookup past analyses matching a draft question. Used by the
    # composer to surface "you asked something like this before" before the
    # user even hits send.
    path("recall", views.recall_lookup, name="recall-lookup"),
    # Slice-3: full body of a single AnalysisNote (the preview shipped in
    # the recall list is truncated to 280 chars).
    path("notes/<int:note_id>", views.analysis_note_detail, name="note-detail"),
]
