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


# ---------------------------------------------------------------------------
# Analytics layer (Slice 1)
#
# After every assistant turn a second LLM pass extracts (company, period,
# metric, value, unit, source) tuples from the answer and upserts them into
# MetricFact. Slice 2 will consult MetricFact before re-running RAG.
# AnalysisNote stores verbatim past answers keyed by analytical scope so
# Slice 3 can surface them proactively. FactProvenance is append-only history
# of every fact write (audit + debugging).
# ---------------------------------------------------------------------------


# Controlled vocabulary. Constraining `unit` to this set stops the extractor
# from inventing units we can't compare across rows ("crores" vs "cr" vs
# "crore" all collapse to inr_crore at normalisation time).
UNIT_CHOICES = [
    ("inr_crore",   "INR crore"),
    ("inr_lakh",    "INR lakh"),
    ("inr",         "INR (absolute)"),
    ("usd_million", "USD million"),
    ("usd",         "USD (absolute)"),
    ("pct",         "Percentage"),
    ("rupees",      "Rupees per share"),
    ("count",       "Count"),
    ("ratio",       "Ratio (x)"),
    ("days",        "Days"),
]

STATEMENT_VARIANT_CHOICES = [
    ("",             "Unspecified"),
    ("standalone",   "Standalone"),
    ("consolidated", "Consolidated"),
]


class MetricFact(models.Model):
    """One canonical metric value for a (company, period, metric_key) tuple.

    UPSERT semantics: the unique constraint covers the dimensions a user
    would treat as 'the same fact' -- consolidated vs standalone are separate
    rows, INR vs USD versions of the same metric are separate rows. A fresh
    extraction with a different value overwrites the row AND snapshots the
    previous version into FactProvenance first."""

    company         = models.CharField(max_length=64, db_index=True)
    period          = models.CharField(max_length=16, db_index=True)  # "Q3FY24" or "FY25"
    metric_key      = models.CharField(max_length=64, db_index=True)  # canonical key from facts.CANONICAL_METRICS
    value           = models.DecimalField(max_digits=20, decimal_places=4)
    unit            = models.CharField(max_length=16, choices=UNIT_CHOICES)
    statement_variant = models.CharField(
        max_length=16, choices=STATEMENT_VARIANT_CHOICES, blank=True, default="",
    )

    # Provenance back to the source PDF page AND the assistant Message that
    # produced this fact (lets Slice 3 re-render the original answer when
    # surfacing a recall).
    source_doc      = models.CharField(max_length=255, blank=True, default="")
    source_page     = models.IntegerField(null=True, blank=True)
    extracted_from  = models.ForeignKey(
        Message, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="extracted_facts",
    )

    # 0-1 self-reported by the extractor LLM. Slice 2's authoritative cache
    # can decline to trust low-confidence rows and re-run RAG instead.
    confidence      = models.FloatField(default=1.0)

    created_at      = models.DateTimeField(auto_now_add=True)
    updated_at      = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["company", "period", "metric_key",
                        "statement_variant", "unit"],
                name="uniq_metric_fact",
            ),
        ]
        indexes = [
            models.Index(fields=["company", "period"]),
            models.Index(fields=["company", "metric_key"]),
        ]
        ordering = ["company", "period", "metric_key"]

    def __str__(self):
        return (f"{self.company} {self.period} {self.metric_key}="
                f"{self.value} {self.unit}")


class FactProvenance(models.Model):
    """Append-only audit log of every MetricFact write.

    operation = "insert"    -> new fact
                "overwrite" -> existing fact's value changed
                "duplicate" -> extractor produced the same value (logged for
                               analytics; does NOT bump MetricFact.updated_at)
    """

    company           = models.CharField(max_length=64, db_index=True)
    period            = models.CharField(max_length=16, db_index=True)
    metric_key        = models.CharField(max_length=64)
    value             = models.DecimalField(max_digits=20, decimal_places=4)
    unit              = models.CharField(max_length=16)
    statement_variant = models.CharField(max_length=16, blank=True, default="")
    source_doc        = models.CharField(max_length=255, blank=True, default="")
    source_page       = models.IntegerField(null=True, blank=True)
    extracted_from    = models.ForeignKey(
        Message, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="provenance_entries",
    )
    confidence        = models.FloatField(default=1.0)
    operation         = models.CharField(max_length=16)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["company", "period", "metric_key", "-created_at"]),
        ]


class AnalysisNote(models.Model):
    """Verbatim assistant answer indexed by its analytical scope.

    Slice 3 will match a new question's (companies, periods, statement)
    against scope and offer a recall + refresh."""

    # Shape:
    #   {"companies": ["infosys"], "periods": ["Q3FY24","Q3FY25"],
    #    "statement": "balance sheet" | "", "metrics": ["revenue", ...]}
    scope          = models.JSONField(default=dict)
    mode           = models.CharField(max_length=20)  # extract|analyze|compare
    body_md        = models.TextField()
    source_message = models.OneToOneField(
        Message, on_delete=models.CASCADE, related_name="analysis_note",
    )
    created_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        scope = self.scope or {}
        cos = ",".join(scope.get("companies") or []) or "?"
        per = ",".join(scope.get("periods") or []) or "?"
        return f"AnalysisNote[{self.mode}] {cos} / {per}"


# ---------------------------------------------------------------------------
# On-the-fly PDF uploads.
#
# A user can attach a PDF to a chat from the composer. The file is indexed
# into its own ChromaDB collection (named UPLOAD_CHROMA_PREFIX + str(id)) and
# stays available for the lifetime of the parent Chat. Deleting the Chat
# cascade-deletes the UploadedDoc row; the cache.py-style cleanup hook in
# uploads.py drops the underlying collection + file from disk.
# ---------------------------------------------------------------------------


class UploadedDoc(models.Model):
    """A PDF uploaded into a single chat and indexed into its own Chroma
    collection. NOT part of the curated `finrag` corpus."""

    STATUS_PENDING = "pending"
    STATUS_INDEXING = "indexing"
    STATUS_READY = "ready"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING,  "Pending"),
        (STATUS_INDEXING, "Indexing"),
        (STATUS_READY,    "Ready"),
        (STATUS_FAILED,   "Failed"),
    ]

    chat            = models.ForeignKey(
        Chat, on_delete=models.CASCADE, related_name="uploads",
    )
    filename        = models.CharField(max_length=255)
    # SHA-256 of file bytes, used for per-chat dedup. Same PDF re-uploaded into
    # the same chat returns the existing row instead of re-indexing.
    sha256          = models.CharField(max_length=64, db_index=True)
    # Stored relative to config.UPLOAD_DIR. Kept on disk so we can re-index
    # after a wipe of the vectorstore if ever needed.
    stored_path     = models.CharField(max_length=512, blank=True, default="")
    pages           = models.IntegerField(default=0)
    chunk_count     = models.IntegerField(default=0)
    collection_name = models.CharField(max_length=128, blank=True, default="")
    status          = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING,
    )
    error           = models.TextField(blank=True, default="")
    created_at      = models.DateTimeField(auto_now_add=True)
    updated_at      = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["chat", "sha256"], name="uniq_chat_upload_sha",
            ),
        ]

    def __str__(self):
        return f"Upload {self.pk}: {self.filename} ({self.status})"
