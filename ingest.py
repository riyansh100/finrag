"""Load PDFs from data/, chunk them, embed with Ollama, persist to ChromaDB."""

import re
import sys
import shutil
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import config


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


def load_pdf(path):
    author = author_from_filename(path.name)
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            header = f"Document: {path.name}\nAuthor: {author}\n\n"
            pages.append(Document(
                page_content=header + text,
                metadata={"source": path.name, "author": author, "page": i + 1},
            ))
    doc.close()
    return pages


def main():
    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.DATA_DIR}. Drop some in and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  - {p.name}")

    all_docs = []
    for path in pdf_paths:
        pages = load_pdf(path)
        print(f"  {path.name}: {len(pages)} non-empty pages")
        all_docs.extend(pages)

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
