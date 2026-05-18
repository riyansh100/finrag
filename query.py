"""RAG query: retrieve from Chroma, answer with Ollama llama3.1.

Phase 2 retrieval:
  - Author-name → source filter (Phase 1, unchanged).
  - Numeric-intent detection → dual retrieval that guarantees table chunks
    appear alongside narrative chunks.
  - Prompt instructs the LLM to cite inline as [source p.N].
"""

import re
import sys
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from ingest import author_from_filename


PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions using only the provided context from project reports and financial documents.

Guidelines:
- Provide a thorough, well-structured answer. Use bullet points or short paragraphs where helpful.
- Cite the source of every concrete fact inline using the format [filename p.N] taken from the chunk headers.
- When quoting numbers from a table, reproduce the relevant row(s) faithfully and cite the table's page.
- If the context does not contain enough information, say so clearly and explain what is missing. Do not invent numbers, names, or dates.

Context:
{context}

Question: {question}

Answer:""")


# --- vectorstore + retriever ------------------------------------------------

def _build_vectorstore():
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )


def _compose_filter(source_filter=None, type_filter=None):
    """Build a Chroma `where` filter combining source + type constraints."""
    clauses = []
    if source_filter:
        clauses.append({"source": source_filter})
    if type_filter:
        clauses.append({"type": type_filter})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _retriever(vectorstore, k, source_filter=None, type_filter=None):
    search_kwargs = {
        "k": k,
        "fetch_k": max(config.MMR_FETCH_K, k * 3),
        "lambda_mult": config.MMR_LAMBDA,
    }
    flt = _compose_filter(source_filter, type_filter)
    if flt:
        search_kwargs["filter"] = flt
    return vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def get_retriever(source_filter=None, type_filter=None):
    """Public single-retriever factory (kept for back-compat with app.py cache)."""
    return _retriever(_build_vectorstore(), k=config.TOP_K,
                      source_filter=source_filter, type_filter=type_filter)


# --- author detection -------------------------------------------------------

def _author_index():
    """Map lowercase name token -> filename, for both first and last names."""
    index = {}
    for path in sorted(config.DATA_DIR.glob("*.pdf")):
        author = author_from_filename(path.name)
        for token in author.split():
            if len(token) >= 3:
                index.setdefault(token.lower(), set()).add(path.name)
    return index


_AUTHOR_INDEX = None


def detect_source_filter(question):
    """Return a filename if the question clearly references one author, else None."""
    global _AUTHOR_INDEX
    if _AUTHOR_INDEX is None:
        _AUTHOR_INDEX = _author_index()
    tokens = re.findall(r"[A-Za-z]+", question.lower())
    matched = set()
    for t in tokens:
        if t in _AUTHOR_INDEX:
            matched |= _AUTHOR_INDEX[t]
    return next(iter(matched)) if len(matched) == 1 else None


# --- numeric-intent detection ----------------------------------------------

_NUMERIC_KEYWORDS = {
    "revenue", "profit", "loss", "income", "expense", "expenses", "cost", "costs",
    "ebitda", "ebit", "margin", "tax", "asset", "assets", "liability", "liabilities",
    "equity", "cash", "debt", "balance", "sheet", "total", "subtotal", "amount",
    "amounts", "value", "values", "price", "prices", "ratio", "ratios", "percent",
    "percentage", "growth", "yoy", "quarter", "fy", "year", "annual",
    "how", "much", "many", "number", "count",
}


def is_numeric_question(question):
    q = question.lower()
    tokens = set(re.findall(r"[a-z]+", q))
    if tokens & _NUMERIC_KEYWORDS:
        return True
    # explicit numbers / currency / years
    if re.search(r"\$|₹|€|£|\bfy\d{2,4}\b|\b(19|20)\d{2}\b|\d[\d,]*\.?\d*", q):
        return True
    return False


# --- context formatting -----------------------------------------------------

def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        kind = d.metadata.get("type", "text")
        tag = "TABLE" if kind == "table" else "TEXT"
        parts.append(f"[{i}] ({src} p.{page}) [{tag}]\n{d.page_content}")
    return "\n\n".join(parts)


def _dedupe(docs):
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"),
               d.metadata.get("type"), d.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# --- main entry -------------------------------------------------------------

def retrieve(question, source_filter=None):
    """Return a list of Documents for the question, applying type-aware retrieval."""
    vectorstore = _build_vectorstore()
    numeric = is_numeric_question(question)

    if numeric:
        k_table = max(1, config.TOP_K // 2)
        k_text = config.TOP_K - k_table
        table_docs = _retriever(vectorstore, k=k_table,
                                source_filter=source_filter,
                                type_filter="table").invoke(question)
        text_docs = _retriever(vectorstore, k=k_text + k_table,
                               source_filter=source_filter).invoke(question)
        docs = _dedupe(table_docs + text_docs)[:config.TOP_K + k_table]
    else:
        docs = _retriever(vectorstore, k=config.TOP_K,
                          source_filter=source_filter).invoke(question)
    return docs, numeric


def ask(question, llm=None):
    source_filter = detect_source_filter(question)
    docs, numeric = retrieve(question, source_filter=source_filter)
    llm = llm or ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    context = format_context(docs)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {
        "answer": answer,
        "sources": docs,
        "filtered_to": source_filter,
        "numeric": numeric,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = ask(q)
    flags = []
    if result["filtered_to"]:
        flags.append(f"filtered to: {result['filtered_to']}")
    if result["numeric"]:
        flags.append("numeric intent → table-biased retrieval")
    if flags:
        print("[" + " | ".join(flags) + "]")
    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources ===")
    for i, d in enumerate(result["sources"], 1):
        kind = d.metadata.get("type", "text")
        print(f"  [{i}] {d.metadata.get('source')} p.{d.metadata.get('page')} [{kind}]")
