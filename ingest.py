"""Load PDFs from data/, chunk them, embed with Ollama, persist to ChromaDB.

Phase 2:
  - Tesseract OCR fallback for scanned / image-only pages.
  - pdfplumber table extraction; tables stored as separate Markdown Documents.
"""

import io
import re
import sys
import shutil
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import make_embeddings, make_vectorstore

try:
    import pytesseract
    from PIL import Image
    _OCR_IMPORTS_OK = True
except ImportError:
    _OCR_IMPORTS_OK = False

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False


def _looks_like_person_report(name):
    """True for filenames like B076_RiyanshSachdev_FPR.pdf."""
    return bool(re.match(r"B\d+_", name))


def author_from_filename(name):
    """B076_RiyanshSachdev_FPR.pdf -> 'Riyansh Sachdev'. Empty string for non-person filenames."""
    if not _looks_like_person_report(name):
        return ""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    parts = [p for p in parts if not re.fullmatch(r"B\d+", p)
             and p.lower() not in {"fpr", "finalreport", "report", "final"}]
    # split camelCase: "RiyanshSachdev" -> "Riyansh Sachdev"
    expanded = []
    for p in parts:
        expanded.extend(re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", p))
    return " ".join(expanded) if expanded else stem


def _chunk_header(filename, author, page=None):
    lines = [f"Document: {filename}"]
    if author:
        lines.append(f"Author: {author}")
    if page is not None:
        lines.append(f"Page: {page}")
    return "\n".join(lines) + "\n\n"


def _ocr_page(page):
    """Render a PyMuPDF page to an image and OCR it with Tesseract."""
    if not _OCR_IMPORTS_OK:
        return ""
    zoom = config.OCR_DPI / 72.0  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang=config.OCR_LANG).strip()
    except pytesseract.TesseractNotFoundError:
        return ""


def _clean_cell(cell):
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _table_to_markdown(rows):
    """Render a list-of-rows table as a GitHub-flavoured Markdown table."""
    rows = [[_clean_cell(c) for c in r] for r in rows if any(c is not None for c in r)]
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * ncols) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_tables(path, skip_pages):
    """Return list of (page_num_1based, markdown) for tables in non-OCR pages."""
    if not (config.TABLE_EXTRACTION_ENABLED and _PDFPLUMBER_OK):
        return []
    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            if page_num in skip_pages:
                continue
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for t in tables:
                if (len(t) < config.TABLE_MIN_ROWS
                        or max((len(r) for r in t), default=0) < config.TABLE_MIN_COLS):
                    continue
                md = _table_to_markdown(t)
                if md:
                    out.append((page_num, md))
    return out


def load_pdf(path):
    author = author_from_filename(path.name)
    doc = fitz.open(path)
    text_docs = []
    ocr_pages = 0
    ocr_page_nums = set()
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        used_ocr = False

        if config.OCR_ENABLED and len(text) < config.OCR_MIN_CHARS:
            ocr_text = _ocr_page(page)
            if len(ocr_text) > len(text):
                text = ocr_text
                used_ocr = True

        if not text:
            continue

        header = _chunk_header(path.name, author)
        text_docs.append(Document(
            page_content=header + text,
            metadata={
                "source": path.name,
                "author": author,
                "page": i + 1,
                "type": "text",
                "ocr": used_ocr,
            },
        ))
        if used_ocr:
            ocr_pages += 1
            ocr_page_nums.add(i + 1)
    doc.close()

    table_docs = []
    for page_num, md in _extract_tables(path, skip_pages=ocr_page_nums):
        header = _chunk_header(path.name, author, page=page_num)
        table_docs.append(Document(
            page_content=header + md,
            metadata={
                "source": path.name,
                "author": author,
                "page": page_num,
                "type": "table",
                "ocr": False,
            },
        ))

    return text_docs, table_docs, ocr_pages


def main():
    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.DATA_DIR}. Drop some in and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  - {p.name}")

    if config.OCR_ENABLED and not _OCR_IMPORTS_OK:
        print("  [warn] OCR enabled but pytesseract / Pillow not installed; "
              "scanned pages will be skipped. Run: pip install pytesseract pillow")
    if config.TABLE_EXTRACTION_ENABLED and not _PDFPLUMBER_OK:
        print("  [warn] Table extraction enabled but pdfplumber not installed; "
              "tables will be flattened into text only. Run: pip install pdfplumber")

    text_docs_all = []
    table_docs_all = []
    total_ocr = 0
    for path in pdf_paths:
        text_docs, table_docs, ocr_count = load_pdf(path)
        bits = [f"{len(text_docs)} text pages"]
        if table_docs:
            bits.append(f"{len(table_docs)} tables")
        if ocr_count:
            bits.append(f"{ocr_count} via OCR")
        print(f"  {path.name}: " + ", ".join(bits))
        text_docs_all.extend(text_docs)
        table_docs_all.extend(table_docs)
        total_ocr += ocr_count
    if total_ocr:
        print(f"\nOCR fallback used on {total_ocr} page(s) total.")
    if table_docs_all:
        print(f"Extracted {len(table_docs_all)} table(s) total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs_all)
    chunks = text_chunks + table_docs_all  # tables stored whole, never split
    print(f"\nSplit into {len(text_chunks)} text chunks + "
          f"{len(table_docs_all)} table chunks = {len(chunks)} total "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = make_embeddings()
    print(f"Embedding + persisting to {config.VECTORSTORE_DIR} ...")
    vectorstore = make_vectorstore(embeddings=embeddings)
    vectorstore.add_documents(chunks)
    print(f"Done. Collection '{config.COLLECTION_NAME}' now has "
          f"{vectorstore._collection.count()} vectors "
          f"(cosine similarity, nomic task-prefixed).")


if __name__ == "__main__":
    main()
