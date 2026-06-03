from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = 'chat'

    def ready(self):
        # Wire the pre_delete signal that drops an upload's Chroma collection
        # + on-disk file when the row goes away (including chat-cascade).
        from . import signals  # noqa: F401
