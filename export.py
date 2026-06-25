"""Chat -> downloadable report (PDF / Markdown / HTML).

A chat is a list of Message rows (see chat.models). This module turns one chat
into a nicely formatted, self-contained report a user can download from the UI.

Three render targets, all driven off the SAME chat object so they stay in sync:

  chat_to_markdown(chat) -> str    plain Markdown (re-importable, zero deps)
  chat_to_html(chat)     -> str    standalone styled HTML doc (print stylesheet)
  chat_to_pdf(chat)      -> bytes  the HTML rendered to PDF via WeasyPrint

Only PDF needs a heavy dependency (WeasyPrint + its pango/cairo system libs),
so it's imported lazily inside chat_to_pdf(): the markdown/html paths work even
on a box where WeasyPrint isn't installed, and the endpoint can surface a clean
error instead of failing at import time.

Per-turn `verification` and live charts are TRANSIENT in this app (returned for
one live turn, never persisted on Message), so they don't appear in an exported
chat. What IS persisted -- question, answer markdown, sources, flags, mode -- is
exactly what the report carries.
"""

from __future__ import annotations

import html as _html
import re

import markdown as _md

# Markdown extensions: GFM-style tables (financial answers are table-heavy),
# fenced code, and sane list handling. Mirrors what `marked` gives the live UI
# (frontend/app.js) so the exported answer matches what the user saw on screen.
_MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "nl2br"]


def _slugify(text: str, fallback: str = "chat") -> str:
    """Filesystem-safe slug for the download filename."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:60] or fallback


def export_filename(chat, ext: str) -> str:
    """`<slugified-title>.<ext>`, e.g. balance-sheet-infosys-q1-fy24.pdf."""
    return f"{_slugify(getattr(chat, 'title', ''), f'chat-{chat.pk}')}.{ext}"


def _render_markdown(text: str) -> str:
    """Answer markdown -> HTML fragment, with tables."""
    return _md.markdown(text or "", extensions=_MD_EXTENSIONS, output_format="html5")


# --- Markdown export --------------------------------------------------------

def chat_to_markdown(chat) -> str:
    """Render a chat as a single Markdown document.

    Near-free relative to the HTML/PDF paths and handy for re-import or pasting
    into a doc. Questions become H2 headings; answers are kept verbatim (they're
    already markdown); sources and flags follow each answer as a compact list.
    """
    lines = [f"# {chat.title or f'Chat {chat.pk}'}", ""]
    lines.append(f"_Exported from FinRAG · {chat.created_at:%Y-%m-%d %H:%M}_")
    lines.append("")

    for msg in chat.messages.all():
        if msg.role == "user":
            lines += [f"## {msg.content.strip()}", ""]
            continue

        # assistant
        if msg.mode:
            lines.append(f"**Mode:** `{msg.mode}`")
        if msg.flags:
            lines.append("**Flags:** " + " · ".join(msg.flags))
        if msg.mode or msg.flags:
            lines.append("")
        lines += [msg.content.strip(), ""]

        if msg.sources:
            lines.append("**Sources**")
            for i, s in enumerate(msg.sources, 1):
                lines.append(
                    f"{i}. {s.get('source', '?')} — p.{s.get('page', '?')}"
                    f" ({s.get('type') or 'text'})"
                )
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# --- HTML / PDF export ------------------------------------------------------

# Standalone print stylesheet. Lives here (not in frontend/style.css) because
# the exported doc must be self-contained -- WeasyPrint gets a single HTML
# string with no access to the app's static files. Light theme on purpose:
# the dark chat UI doesn't print well, and a report is a white-paper artefact.
_REPORT_CSS = """
@page {
  size: A4;
  margin: 22mm 18mm 20mm 18mm;
  @bottom-center {
    content: "FinRAG report · page " counter(page) " of " counter(pages);
    font: 8pt 'Helvetica Neue', Arial, sans-serif;
    color: #99a;
  }
}
body {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1c2430;
}
.cover { border-bottom: 2px solid #2563eb; padding-bottom: 10px; margin-bottom: 22px; }
.cover h1 { font-size: 20pt; margin: 0 0 4px; color: #0f172a; }
.cover .meta { font-size: 9pt; color: #64748b; }
.turn { margin: 0 0 20px; page-break-inside: avoid; }
.question {
  background: #eff4ff; border-left: 3px solid #2563eb;
  padding: 8px 12px; border-radius: 0 4px 4px 0;
  font-weight: 600; color: #0f172a;
}
.answer { margin-top: 10px; }
.answer table {
  border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt;
}
.answer th, .answer td {
  border: 1px solid #d6deea; padding: 5px 8px; text-align: left;
}
.answer th { background: #f1f5f9; }
.answer td:nth-child(n+2) { text-align: right; font-variant-numeric: tabular-nums; }
.answer code, .answer pre {
  font-family: 'SFMono-Regular', Consolas, monospace; font-size: 9pt;
}
.answer pre { background: #f6f8fa; padding: 8px; border-radius: 4px; overflow-x: auto; }
.badges { margin-top: 8px; }
.badge {
  display: inline-block; font-size: 8pt; padding: 1px 7px; border-radius: 10px;
  margin-right: 5px; background: #e2e8f0; color: #475569;
}
.badge.mode { background: #2563eb; color: #fff; text-transform: uppercase; }
.sources { margin-top: 10px; font-size: 8.5pt; color: #475569; }
.sources .head { font-weight: 600; color: #334155; margin-bottom: 3px; }
.sources li { margin: 1px 0; }
hr.sep { border: none; border-top: 1px solid #e2e8f0; margin: 18px 0; }
"""


def chat_to_html(chat) -> str:
    """Render a chat as a standalone, styled HTML document (string).

    Used directly for the ?format=html download and as the input to
    chat_to_pdf(). Everything (CSS included) is inlined so the file opens
    correctly with no server or network.
    """
    title = _html.escape(chat.title or f"Chat {chat.pk}")
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{title}</title><style>{_REPORT_CSS}</style></head><body>",
        "<div class='cover'>",
        f"<h1>{title}</h1>",
        f"<div class='meta'>FinRAG report · generated {chat.created_at:%Y-%m-%d %H:%M}"
        f" · {chat.messages.count()} messages</div>",
        "</div>",
    ]

    for msg in chat.messages.all():
        if msg.role == "user":
            parts.append(
                f"<div class='turn'><div class='question'>"
                f"{_html.escape(msg.content)}</div>"
            )
            # Answer (if any) is appended by the following assistant turn; close
            # the turn div here and let the assistant open its own.
            parts.append("</div>")
            continue

        # assistant
        parts.append("<div class='turn'>")
        parts.append(f"<div class='answer'>{_render_markdown(msg.content)}</div>")

        badges = []
        if msg.mode:
            badges.append(f"<span class='badge mode'>{_html.escape(msg.mode)}</span>")
        for flag in msg.flags or []:
            badges.append(f"<span class='badge'>{_html.escape(str(flag))}</span>")
        if badges:
            parts.append("<div class='badges'>" + "".join(badges) + "</div>")

        if msg.sources:
            parts.append("<div class='sources'><div class='head'>Sources</div><ol>")
            for s in msg.sources:
                src = _html.escape(str(s.get("source", "?")))
                page = _html.escape(str(s.get("page", "?")))
                typ = _html.escape(str(s.get("type") or "text"))
                parts.append(f"<li>{src} — p.{page} ({typ})</li>")
            parts.append("</ol></div>")

        parts.append("</div><hr class='sep'>")

    parts.append("</body></html>")
    return "".join(parts)


def chat_to_pdf(chat) -> bytes:
    """Render a chat to PDF bytes via WeasyPrint.

    WeasyPrint is imported lazily: it pulls heavy system libs (pango/cairo), so
    only this path requires them. Raises RuntimeError with an actionable message
    when WeasyPrint isn't installed, which the view turns into a clean 503.
    """
    try:
        from weasyprint import HTML
    except Exception as e:  # ImportError, or missing system libs at import time
        raise RuntimeError(
            "PDF export needs WeasyPrint (and its pango/cairo system libs). "
            "Install `weasyprint` and the OS deps, or use ?format=md / ?format=html."
        ) from e
    return HTML(string=chat_to_html(chat)).write_pdf()
