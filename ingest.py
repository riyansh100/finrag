"""Load PDFs from data/, chunk them, embed with Ollama, persist to ChromaDB.

Pipeline:
  - PyMuPDF text extraction, with Tesseract OCR fallback for image-only pages.
  - Table extraction per page:
      * native pages: pdfplumber (ruled tables) → Camelot stream (borderless fallback)
      * OCR'd pages:  img2table (re-uses the rendered pixmap we already produced)
  - Tables emitted as separate Markdown Documents (type="table"), never chunked.
"""

import io
import re
import sys
import warnings
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

try:
    import camelot
    _CAMELOT_OK = True
except ImportError:
    _CAMELOT_OK = False

try:
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    _IMG2TABLE_OK = True
    _IMG2TABLE_OCR = None  # built lazily
except ImportError:
    _IMG2TABLE_OK = False
    _IMG2TABLE_OCR = None


# --- filename / header helpers ---------------------------------------------

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


# --- OCR -------------------------------------------------------------------

def _render_page_image(page):
    """Render a PyMuPDF page to a PIL Image at OCR_DPI. Returns (PIL.Image, png_bytes)."""
    zoom = config.OCR_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(png_bytes))
    return img, png_bytes


def _ocr_image(img):
    if not _OCR_IMPORTS_OK:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=config.OCR_LANG).strip()
    except pytesseract.TesseractNotFoundError:
        return ""


# --- table formatting ------------------------------------------------------

def _clean_cell(cell):
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _table_to_markdown(rows):
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


def _table_passes_filter(rows):
    return (len(rows) >= config.TABLE_MIN_ROWS
            and max((len(r) for r in rows), default=0) >= config.TABLE_MIN_COLS)


# --- per-page table extractors ---------------------------------------------

def _extract_plumber(plumber_page):
    """Ruled tables from pdfplumber. Returns list of markdown strings."""
    if not _PDFPLUMBER_OK:
        return []
    try:
        tables = plumber_page.extract_tables() or []
    except Exception:
        tables = []
    out = []
    for t in tables:
        if _table_passes_filter(t):
            md = _table_to_markdown(t)
            if md:
                out.append(md)
    return out


def _extract_camelot_stream(path, page_num):
    """Camelot stream-flavour for borderless / whitespace-aligned tables."""
    if not (_CAMELOT_OK and config.TABLE_STREAM_FALLBACK_ENABLED):
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = camelot.read_pdf(str(path), pages=str(page_num), flavor="stream")
    except Exception:
        return []
    out = []
    for t in tables:
        rows = t.df.values.tolist()
        # Camelot frequently includes empty leading rows; trim.
        rows = [r for r in rows if any(_clean_cell(c) for c in r)]
        if _table_passes_filter(rows):
            md = _table_to_markdown(rows)
            if md:
                out.append(md)
    return out


def _get_img2table_ocr():
    global _IMG2TABLE_OCR
    if _IMG2TABLE_OCR is None and _IMG2TABLE_OK:
        _IMG2TABLE_OCR = TesseractOCR(n_threads=1, lang=config.OCR_LANG)
    return _IMG2TABLE_OCR


def _extract_img2table(png_bytes):
    """Detect + OCR tables from a page image (for scanned/OCR'd pages)."""
    if not (_IMG2TABLE_OK and config.TABLE_OCR_ENABLED):
        return []
    try:
        doc = Img2TableImage(src=io.BytesIO(png_bytes))
        extracted = doc.extract_tables(ocr=_get_img2table_ocr(), implicit_rows=True,
                                       borderless_tables=True, min_confidence=50)
    except Exception:
        return []
    out = []
    for tbl in extracted:
        df = tbl.df
        if df is None or df.empty:
            continue
        rows = df.fillna("").astype(str).values.tolist()
        if _table_passes_filter(rows):
            md = _table_to_markdown(rows)
            if md:
                out.append(md)
    return out


# --- main per-PDF loader ---------------------------------------------------

def load_pdf(path):
    author = author_from_filename(path.name)
    doc = fitz.open(path)

    # Open pdfplumber for this PDF once (closed at the end). Tolerate failure.
    try:
        plumber = pdfplumber.open(path) if _PDFPLUMBER_OK else None
    except Exception:
        plumber = None

    text_docs = []
    table_docs = []
    counts = {"ocr": 0, "plumber": 0, "stream": 0, "img2table": 0}

    for i, page in enumerate(doc):
        page_num = i + 1
        text = page.get_text().strip()
        used_ocr = False
        png_bytes = None

        if config.OCR_ENABLED and len(text) < config.OCR_MIN_CHARS:
            img, png_bytes = _render_page_image(page)
            ocr_text = _ocr_image(img)
            if len(ocr_text) > len(text):
                text = ocr_text
                used_ocr = True

        if text:
            text_docs.append(Document(
                page_content=_chunk_header(path.name, author) + text,
                metadata={"source": path.name, "author": author,
                          "page": page_num, "type": "text", "ocr": used_ocr},
            ))
            if used_ocr:
                counts["ocr"] += 1

        if not config.TABLE_EXTRACTION_ENABLED:
            continue

        page_tables = []  # list of (markdown, engine)
        if used_ocr:
            if png_bytes is None:
                _, png_bytes = _render_page_image(page)
            for md in _extract_img2table(png_bytes):
                page_tables.append((md, "img2table"))
        else:
            mds = _extract_plumber(plumber.pages[i]) if plumber else []
            engine = "plumber"
            if not mds:
                mds = _extract_camelot_stream(path, page_num)
                engine = "stream"
            for md in mds:
                page_tables.append((md, engine))

        for md, engine in page_tables:
            counts[engine] += 1
            table_docs.append(Document(
                page_content=_chunk_header(path.name, author, page=page_num) + md,
                metadata={"source": path.name, "author": author,
                          "page": page_num, "type": "table",
                          "ocr": used_ocr, "table_engine": engine},
            ))

    if plumber is not None:
        plumber.close()
    doc.close()
    return text_docs, table_docs, counts


# --- driver ----------------------------------------------------------------

def main():
    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.DATA_DIR}. Drop some in and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  - {p.name}")

    if config.OCR_ENABLED and not _OCR_IMPORTS_OK:
        print("  [warn] OCR enabled but pytesseract/Pillow missing.")
    if config.TABLE_EXTRACTION_ENABLED and not _PDFPLUMBER_OK:
        print("  [warn] pdfplumber missing; ruled-table extraction disabled.")
    if config.TABLE_STREAM_FALLBACK_ENABLED and not _CAMELOT_OK:
        print("  [warn] camelot-py missing; borderless-table fallback disabled.")
    if config.TABLE_OCR_ENABLED and not _IMG2TABLE_OK:
        print("  [warn] img2table missing; tables on scanned pages will be missed.")

    text_docs_all = []
    table_docs_all = []
    totals = {"ocr": 0, "plumber": 0, "stream": 0, "img2table": 0}
    for path in pdf_paths:
        text_docs, table_docs, counts = load_pdf(path)
        bits = [f"{len(text_docs)} text pages"]
        if counts["ocr"]:
            bits.append(f"{counts['ocr']} via OCR")
        table_bits = []
        for engine in ("plumber", "stream", "img2table"):
            if counts[engine]:
                table_bits.append(f"{counts[engine]} {engine}")
        if table_bits:
            bits.append(f"tables: {', '.join(table_bits)}")
        print(f"  {path.name}: " + "; ".join(bits))
        text_docs_all.extend(text_docs)
        table_docs_all.extend(table_docs)
        for k in totals:
            totals[k] += counts[k]

    print()
    print(f"OCR fallback: {totals['ocr']} page(s).")
    print(f"Tables: {totals['plumber']} pdfplumber + "
          f"{totals['stream']} camelot-stream + "
          f"{totals['img2table']} img2table "
          f"= {len(table_docs_all)} total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs_all)
    chunks = text_chunks + table_docs_all
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
