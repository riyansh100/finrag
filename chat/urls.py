"""URL routes for the chat app, mounted under /api/ by the project urlconf."""

from django.urls import path

from . import views

urlpatterns = [
    path("chats", views.chat_list, name="chat-list"),
    path("chats/<int:chat_id>", views.chat_detail, name="chat-detail"),
    path("chats/<int:chat_id>/messages", views.message_create, name="message-create"),
    path("modes", views.mode_list, name="mode-list"),
]
