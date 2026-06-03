"""Lifecycle hooks for the chat app.

Right now: when an UploadedDoc row is deleted (directly or via Chat cascade),
tear down its Chroma collection + stored PDF. Keeping this in a signal rather
than the view means cascade deletes from `chat.delete()` also clean up — DRF's
delete path doesn't go through chat_upload_detail.
"""

from django.db.models.signals import pre_delete
from django.dispatch import receiver

import uploads as upload_store

from .models import UploadedDoc


@receiver(pre_delete, sender=UploadedDoc)
def _drop_upload_artifacts(sender, instance, **_kwargs):
    upload_store.drop_upload(instance)
