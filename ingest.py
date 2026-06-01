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
from parsers import PARSERS, parse_filename

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

try:
    import base64
    import ollama
    _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False


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


def _chunk_header(filename, author, page=None, meta=None):
    """Top-of-chunk header embedded INTO the text so it's part of the vector.
    `meta` (from parsers.parse_filename) adds Company/Period lines for quarterlies
    and annuals so the LLM can attribute facts to the right period."""
    lines = [f"Document: {filename}"]
    if meta:
        if meta.get("company"):
            lines.append(f"Company: {meta['company'].title()}")
        if meta.get("period"):
            lines.append(f"Period: {meta['period']}")
    if author:
        lines.append(f"Author: {author}")
    if page is not None:
        lines.append(f"Page: {page}")
    return "\n".join(lines) + "\n\n"


# --- OCR -------------------------------------------------------------------

def _render_page_image(page, dpi=None):
    """Render a PyMuPDF page to a PIL Image. Returns (PIL.Image, png_bytes)."""
    dpi = dpi or config.OCR_DPI
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(png_bytes))
    return img, png_bytes


def _page_has_significant_image(page, min_px=None):
    """True if the page has at least one embedded image >= min_px on either side."""
    min_px = min_px or config.FIGURE_MIN_IMAGE_PX
    try:
        images = page.get_images(full=False)
    except Exception:
        return False
    for img_ref in images:
        # img_ref tuple: (xref, smask, w, h, bpc, colorspace, ...)
        w, h = img_ref[2], img_ref[3]
        if w >= min_px or h >= min_px:
            return True
    return False


def _describe_page_with_vision(page):
    """Run the vision model over a rendered page; return description text or empty string."""
    if not (config.FIGURE_DESCRIPTIONS_ENABLED and _OLLAMA_OK):
        return ""
    _, png_bytes = _render_page_image(page, dpi=config.VISION_DPI)
    try:
        response = ollama.chat(
            model=config.VISION_MODEL,
            messages=[{
                "role": "user",
                "content": config.FIGURE_PROMPT,
                "images": [base64.b64encode(png_bytes).decode("ascii")],
            }],
            options={"temperature": 0.1},
        )
        return (response.get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"    [warn] vision call failed: {type(e).__name__}: {e}")
        return ""


def _ocr_image(img):
    """Best-effort OCR. Returns "" on ANY failure (binary, missing binary,
    unicode decode error in pytesseract's stderr parser, etc.) — one bad page
    must never abort a multi-PDF ingest."""
    if not _OCR_IMPORTS_OK:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=config.OCR_LANG).strip()
    except Exception as e:
        print(f"    [warn] OCR failed on page: {type(e).__name__}: {str(e)[:120]}")
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


_STATEMENT_RE = re.compile(
    r"((?:standalone|consolidated)?\s*"
    r"(?:statement of profit and loss|balance sheet|statement of cash flow|"
    r"cash flow statement|statement of changes in equity|"
    r"statement of profit & loss))",
    re.IGNORECASE,
)


def _statement_title(page_text):
    """Extract a financial-statement title from page text so table chunks carry
    their section label (standalone vs consolidated, which statement). Returns
    the best title line, or '' if none found."""
    matches = []
    for line in page_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _STATEMENT_RE.search(line)
        if m:
            matches.append(line)
    if not matches:
        return ""
    # Prefer a line that names standalone/consolidated explicitly.
    for line in matches:
        low = line.lower()
        if "standalone" in low or "consolidated" in low:
            return line
    return matches[0]


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

def load_pdf(path, doc_meta=None):
    """Load one PDF.

    doc_meta: optional dict from parsers.parse_filename — merged into every
              chunk's metadata (company/period/quarter/fy/doc_type). Pass None
              for one-off files outside the per-company layout.
    """
    author = author_from_filename(path.name)
    doc_meta = doc_meta or {}
    doc = fitz.open(path)

    # Open pdfplumber for this PDF once (closed at the end). Tolerate failure.
    try:
        plumber = pdfplumber.open(path) if _PDFPLUMBER_OK else None
    except Exception:
        plumber = None

    text_docs = []
    table_docs = []
    figure_docs = []
    counts = {"ocr": 0, "plumber": 0, "stream": 0, "img2table": 0, "figure": 0}

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
                page_content=_chunk_header(path.name, author, meta=doc_meta) + text,
                metadata={"source": path.name, "author": author,
                          "page": page_num, "type": "text", "ocr": used_ocr,
                          **doc_meta},
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

        # Section label so a table chunk is self-describing (e.g. a balance
        # sheet table knows it's "Consolidated" vs "Standalone").
        title = _statement_title(text)
        title_line = f"Section: {title}\n" if title else ""

        for md, engine in page_tables:
            counts[engine] += 1
            table_docs.append(Document(
                page_content=_chunk_header(path.name, author, page=page_num,
                                           meta=doc_meta)
                             + title_line + md,
                metadata={"source": path.name, "author": author,
                          "page": page_num, "type": "table",
                          "ocr": used_ocr, "table_engine": engine,
                          "section": title, **doc_meta},
            ))

        # Figure description (native pages only — OCR'd pages ARE images and
        # would all match; skip them to avoid noise).
        if (config.FIGURE_DESCRIPTIONS_ENABLED
                and not used_ocr
                and _page_has_significant_image(page)):
            desc = _describe_page_with_vision(page)
            if desc:
                figure_docs.append(Document(
                    page_content=_chunk_header(path.name, author, page=page_num,
                                               meta=doc_meta)
                                 + "Figure description:\n" + desc,
                    metadata={"source": path.name, "author": author,
                              "page": page_num, "type": "figure",
                              "ocr": False, "vision_model": config.VISION_MODEL,
                              **doc_meta},
                ))
                counts["figure"] += 1

    if plumber is not None:
        plumber.close()
    doc.close()
    return text_docs, table_docs, figure_docs, counts


# --- driver ----------------------------------------------------------------

def _discover_pdfs():
    """Yield (path, company_folder, parsed_meta) for every PDF in data/.

    Layout: data/<company>/*.pdf — one folder per company, parser registered
    in parsers.PARSERS. Folders prefixed with "_" are skipped (e.g. _archive).
    Stray PDFs at the top level of data/ are still ingested (no company meta).
    """
    items = []
    for child in sorted(config.DATA_DIR.iterdir()):
        if child.is_file() and child.suffix.lower() == ".pdf":
            items.append((child, None, {}))
            continue
        if not child.is_dir() or child.name.startswith("_"):
            continue
        company = child.name
        for pdf in sorted(child.glob("*.pdf")):
            meta = parse_filename(company, pdf.name) or {}
            items.append((pdf, company, meta))
    return items


def main():
    items = _discover_pdfs()
    if not items:
        print(f"No PDFs found in {config.DATA_DIR}. Drop some in and re-run.")
        sys.exit(1)

    print(f"Found {len(items)} PDF(s):")
    for path, company, meta in items:
        tag = f"  [{company}]" if company else "  [<root>]"
        period = f" period={meta.get('period')}" if meta.get("period") else ""
        print(f"{tag} {path.name}{period}")
    if not any(company in PARSERS for _, company, _ in items):
        print("  [warn] No PDFs matched a registered company folder; "
              "add one to parsers.PARSERS to enable period metadata.")

    if config.OCR_ENABLED and not _OCR_IMPORTS_OK:
        print("  [warn] OCR enabled but pytesseract/Pillow missing.")
    if config.TABLE_EXTRACTION_ENABLED and not _PDFPLUMBER_OK:
        print("  [warn] pdfplumber missing; ruled-table extraction disabled.")
    if config.TABLE_STREAM_FALLBACK_ENABLED and not _CAMELOT_OK:
        print("  [warn] camelot-py missing; borderless-table fallback disabled.")
    if config.TABLE_OCR_ENABLED and not _IMG2TABLE_OK:
        print("  [warn] img2table missing; tables on scanned pages will be missed.")
    if config.FIGURE_DESCRIPTIONS_ENABLED and not _OLLAMA_OK:
        print("  [warn] ollama python pkg missing; figure descriptions disabled.")

    text_docs_all = []
    table_docs_all = []
    figure_docs_all = []
    totals = {"ocr": 0, "plumber": 0, "stream": 0, "img2table": 0, "figure": 0}
    for path, _company, meta in items:
        text_docs, table_docs, figure_docs, counts = load_pdf(path, doc_meta=meta)
        bits = [f"{len(text_docs)} text pages"]
        if counts["ocr"]:
            bits.append(f"{counts['ocr']} via OCR")
        table_bits = []
        for engine in ("plumber", "stream", "img2table"):
            if counts[engine]:
                table_bits.append(f"{counts[engine]} {engine}")
        if table_bits:
            bits.append(f"tables: {', '.join(table_bits)}")
        if counts["figure"]:
            bits.append(f"{counts['figure']} figures")
        print(f"  {path.name}: " + "; ".join(bits))
        text_docs_all.extend(text_docs)
        table_docs_all.extend(table_docs)
        figure_docs_all.extend(figure_docs)
        for k in totals:
            totals[k] += counts[k]

    print()
    print(f"OCR fallback: {totals['ocr']} page(s).")
    print(f"Tables: {totals['plumber']} pdfplumber + "
          f"{totals['stream']} camelot-stream + "
          f"{totals['img2table']} img2table "
          f"= {len(table_docs_all)} total.")
    print(f"Figures described: {len(figure_docs_all)} "
          f"(model: {config.VISION_MODEL}).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs_all)
    chunks = text_chunks + table_docs_all + figure_docs_all
    print(f"\nSplit into {len(text_chunks)} text chunks + "
          f"{len(table_docs_all)} table chunks + "
          f"{len(figure_docs_all)} figure chunks = {len(chunks)} total "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = make_embeddings()
    print(f"Embedding + persisting to {config.VECTORSTORE_DIR} ...")
    vectorstore = make_vectorstore(embeddings=embeddings)
    # Add in small batches so a transient Ollama hiccup mid-corpus doesn't
    # discard everything that came before, and so we get visible progress.
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        try:
            vectorstore.add_documents(batch)
            print(f"  embedded {i + len(batch)}/{len(chunks)} "
                  f"(total in DB: {vectorstore._collection.count()})",
                  flush=True)
        except Exception as e:
            print(f"  [warn] batch {i}-{i + len(batch)} failed: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
    print(f"Done. Collection '{config.COLLECTION_NAME}' now has "
          f"{vectorstore._collection.count()} vectors "
          f"(cosine similarity, nomic task-prefixed).")


if __name__ == "__main__":
    main()
