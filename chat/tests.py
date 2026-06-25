"""Tests for chat export (PDF / Markdown / HTML report download).

These exercise the /api/chats/{id}/export endpoint and the export.py builders.
They need no LLM, no vector index, and no Ollama -- just the DB -- so they run
in CI alongside the deterministic eval slice.
"""

from django.test import TestCase, override_settings
from django.urls import reverse

import export
from .models import Chat, Message


class ExportTestBase(TestCase):
    def setUp(self):
        self.chat = Chat.objects.create(title="Balance sheet — Infosys Q1 FY24")
        Message.objects.create(
            chat=self.chat, role=Message.USER,
            content="give me the balance sheet for infosys Q1 FY24",
        )
        Message.objects.create(
            chat=self.chat, role=Message.ASSISTANT, mode="extract",
            content=(
                "## Balance Sheet — Infosys, Q1 FY24\n\n"
                "| Item | Value |\n| --- | --- |\n"
                "| Total assets | 1,31,322 |\n"
            ),
            sources=[{"source": "q1-2024.pdf", "page": 4, "type": "table",
                      "content": "Total assets 131322"}],
            flags=["Filtered to: infosys · Q1FY24"],
        )


class ExportBuildersTest(ExportTestBase):
    def test_markdown_contains_question_answer_and_sources(self):
        md = export.chat_to_markdown(self.chat)
        self.assertIn("give me the balance sheet for infosys Q1 FY24", md)
        self.assertIn("Total assets", md)
        self.assertIn("q1-2024.pdf", md)
        self.assertIn("extract", md)

    def test_html_is_standalone_and_renders_table(self):
        doc = export.chat_to_html(self.chat)
        self.assertTrue(doc.lstrip().startswith("<!DOCTYPE html>"))
        # Markdown table -> real HTML table in the answer.
        self.assertIn("<table>", doc)
        self.assertIn("1,31,322", doc)
        self.assertIn("q1-2024.pdf", doc)

    def test_html_escapes_user_content(self):
        Message.objects.create(
            chat=self.chat, role=Message.USER, content="<script>x</script>",
        )
        doc = export.chat_to_html(self.chat)
        self.assertNotIn("<script>x</script>", doc)
        self.assertIn("&lt;script&gt;", doc)

    def test_filename_is_slugified(self):
        self.assertEqual(
            export.export_filename(self.chat, "pdf"),
            "balance-sheet-infosys-q1-fy24.pdf",
        )


# Settings populate ALLOWED_HOSTS, so Django won't auto-add the test client's
# 'testserver' host. Allow it explicitly for the HTTP-level tests.
@override_settings(ALLOWED_HOSTS=["testserver"])
class ExportEndpointTest(ExportTestBase):
    def _url(self, fmt):
        return reverse("chat-export", args=[self.chat.pk]) + f"?fmt={fmt}"

    def test_markdown_download(self):
        res = self.client.get(self._url("md"))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["Content-Type"], "text/markdown; charset=utf-8")
        self.assertIn("attachment", res["Content-Disposition"])
        self.assertIn(".md", res["Content-Disposition"])
        self.assertIn(b"Total assets", res.content)

    def test_html_download(self):
        res = self.client.get(self._url("html"))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["Content-Type"], "text/html; charset=utf-8")
        self.assertIn(b"<table>", res.content)

    def test_unknown_format_is_400(self):
        res = self.client.get(self._url("docx"))
        self.assertEqual(res.status_code, 400)

    def test_missing_chat_is_404(self):
        res = self.client.get(reverse("chat-export", args=[99999]))
        self.assertEqual(res.status_code, 404)

    def test_pdf_download_or_clean_503(self):
        """PDF works when WeasyPrint + system libs are present; otherwise the
        endpoint must degrade to a clean 503 (never a 500)."""
        res = self.client.get(self._url("pdf"))
        self.assertIn(res.status_code, (200, 503))
        if res.status_code == 200:
            self.assertEqual(res["Content-Type"], "application/pdf")
            self.assertTrue(res.content.startswith(b"%PDF"))
            self.assertIn("attachment", res["Content-Disposition"])
        else:
            self.assertIn("WeasyPrint", res.json()["detail"])
