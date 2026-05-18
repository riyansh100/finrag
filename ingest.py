"""Load PDFs from data/, chunk them, embed with Ollama, persist to ChromaDB.

Phase 2 adds Tesseract OCR fallback for scanned / image-only pages.
"""

import io
import re
import sys
import shutil
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import config

try:
    import pytesseract
    from PIL import Image
    _OCR_IMPORTS_OK = True
except ImportError:
    _OCR_IMPORTS_OK = False


def author_from_filename(name):
    """B076_RiyanshSachdev_FPR.pdf -> 'Riyansh Sachdev'."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    parts = [p for p in parts if not re.fullmatch(r"B\d+", p)
             and p.lower() not in {"fpr", "finalreport", "report", "final"}]
    # split camelCase: "RiyanshSachdev" -> "Riyansh Sachdev"
    expanded = []
    for p in parts:
        expanded.extend(re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", p))
    return " ".join(expanded) if expanded else stem


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


def load_pdf(path):
    author = author_from_filename(path.name)
    doc = fitz.open(path)
    pages = []
    ocr_pages = 0
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

        header = f"Document: {path.name}\nAuthor: {author}\n\n"
        pages.append(Document(
            page_content=header + text,
            metadata={
                "source": path.name,
                "author": author,
                "page": i + 1,
                "ocr": used_ocr,
            },
        ))
        if used_ocr:
            ocr_pages += 1

    doc.close()
    return pages, ocr_pages


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

    all_docs = []
    total_ocr = 0
    for path in pdf_paths:
        pages, ocr_count = load_pdf(path)
        suffix = f" ({ocr_count} via OCR)" if ocr_count else ""
        print(f"  {path.name}: {len(pages)} non-empty pages{suffix}")
        all_docs.extend(pages)
        total_ocr += ocr_count
    if total_ocr:
        print(f"\nOCR fallback used on {total_ocr} page(s) total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\nSplit into {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    print(f"Embedding + persisting to {config.VECTORSTORE_DIR} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    print(f"Done. Collection '{config.COLLECTION_NAME}' now has "
          f"{vectorstore._collection.count()} vectors.")


if __name__ == "__main__":
    main()
