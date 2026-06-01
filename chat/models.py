"""Database models for the chat app.

Two tables:
  Chat    - one conversation (the sidebar entries).
  Message - one turn in a conversation (user or assistant), linked to its Chat.

This is the persistent replacement for Streamlit's in-RAM
st.session_state.history: chats and messages now survive restarts and there
can be many saved chats at once.
"""

from django.db import models


class Chat(models.Model):
    """A single conversation."""

    # Shown in the sidebar. Auto-derived from the first question (in the view),
    # but stored as a real column so we can support rename later.
    title = models.CharField(max_length=200, default="New chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  # bumped on each new message

    class Meta:
        # Most recently active chat first -> natural sidebar ordering.
        ordering = ["-updated_at"]

    def __str__(self):
        return f"Chat {self.pk}: {self.title}"


class Message(models.Model):
    """One turn in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    ROLE_CHOICES = [(USER, "user"), (ASSISTANT, "assistant")]

    # Link back to the parent chat. on_delete=CASCADE => deleting a chat deletes
    # its messages. related_name="messages" lets us do chat.messages.all().
    chat = models.ForeignKey(
        Chat, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()

    # Assistant turns carry retrieval metadata so the frontend can render the
    # Sources panel and flag captions exactly like the Streamlit UI did.
    # sources: list of {source, page, type, content} dicts.
    sources = models.JSONField(default=list, blank=True)
    # flags: strings like "Rewrote -> ...", "Filtered to: ...", "Numeric intent".
    flags = models.JSONField(default=list, blank=True)
    # The mode this turn was answered in ("extract" | "analyze" | "compare").
    # Blank for user turns; set for assistants and saved with the message so a
    # chat can show per-turn mode badges and the composer can default to "the
    # last mode used in this chat".
    mode = models.CharField(max_length=20, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Chronological within a chat -> the order we replay turns in.
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role} @ chat {self.chat_id}: {self.content[:40]}"
