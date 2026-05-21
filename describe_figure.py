"""Describe a figure / page of a PDF using a vision LLM via Ollama.

Usage:
    python describe_figure.py <filename-or-substring> <page>
    python describe_figure.py B076_RiyanshSachdev_FPR.pdf 16
    python describe_figure.py riyansh 16
    python describe_figure.py riyansh 16 --prompt "What does this architecture diagram show?"

Renders the page at OCR_DPI, encodes it as PNG, and sends it to VISION_MODEL
(default: moondream). Requires `ollama pull <VISION_MODEL>` first.
"""

import argparse
import base64
import sys
from pathlib import Path

import fitz  # PyMuPDF
import ollama

import config


DEFAULT_PROMPT = (
    "Describe this page in detail. If it contains a diagram, flowchart, or "
    "architecture figure, explain every box, label, arrow, and the overall "
    "flow. If it contains a table, transcribe it. Be thorough and concrete."
)


def resolve_pdf(query):
    """Match `query` against filenames in data/. Returns Path or exits."""
    data_dir = config.DATA_DIR
    candidates = list(data_dir.glob("*.pdf"))
    # exact match first
    for p in candidates:
        if p.name == query:
            return p
    # case-insensitive substring
    q = query.lower()
    matches = [p for p in candidates if q in p.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"No PDF in {data_dir} matches {query!r}.\n"
                 f"Available: {[p.name for p in candidates]}")
    sys.exit(f"Ambiguous: {query!r} matches multiple PDFs: "
             f"{[p.name for p in matches]}")


VISION_DPI = 144  # vision models downsample internally; 144 is plenty and ~4x smaller than 300


def render_page_png(path, page_num, dpi=VISION_DPI):
    doc = fitz.open(path)
    if page_num < 1 or page_num > len(doc):
        sys.exit(f"Page {page_num} out of range; {path.name} has {len(doc)} pages.")
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    png = pix.tobytes("png")
    doc.close()
    return png


def describe(path, page_num, prompt, model):
    png = render_page_png(path, page_num)
    print(f"→ Rendering {path.name} p.{page_num} at {VISION_DPI} DPI "
          f"({len(png)//1024} KB) and sending to {model}...\n")
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(png).decode("ascii")],
        }],
        options={"temperature": 0.1},
    )
    print(response["message"]["content"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Filename or substring to match a PDF in data/")
    parser.add_argument("page", type=int, help="1-based page number")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt sent to the vision model")
    parser.add_argument("--model", default=config.VISION_MODEL,
                        help=f"Ollama vision model (default: {config.VISION_MODEL})")
    args = parser.parse_args()

    pdf = resolve_pdf(args.pdf)
    describe(pdf, args.page, args.prompt, args.model)


if __name__ == "__main__":
    main()
