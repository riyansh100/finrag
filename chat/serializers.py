"""DRF serializers: convert model rows <-> JSON.

A ModelSerializer reads the model definition and auto-builds the field list,
so we only declare what differs from the defaults.
"""

from rest_framework import serializers

from .models import Chat, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "sources", "flags", "created_at"]
        # Messages are created by the message endpoint (Step A5), never written
        # directly through this serializer.
        read_only_fields = fields


class ChatSerializer(serializers.ModelSerializer):
    """Sidebar / list representation: no messages, plus a cheap message count."""

    message_count = serializers.IntegerField(source="messages.count", read_only=True)

    class Meta:
        model = Chat
        fields = ["id", "title", "created_at", "updated_at", "message_count"]
        read_only_fields = ["id", "created_at", "updated_at", "message_count"]


class ChatDetailSerializer(ChatSerializer):
    """Single-chat representation: includes the full message list."""

    messages = MessageSerializer(many=True, read_only=True)

    class Meta(ChatSerializer.Meta):
        fields = ChatSerializer.Meta.fields + ["messages"]
