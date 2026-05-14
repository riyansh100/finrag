"""RAG query: retrieve from Chroma, answer with Ollama llama3.1."""

import re
import sys
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from ingest import author_from_filename


PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions using only the provided context from project reports.
Provide a thorough, well-structured answer drawing on relevant details from the context. Use multiple paragraphs or bullet points when appropriate.
If the context does not contain enough information to answer, say so clearly and explain what is missing.

Context:
{context}

Question: {question}

Answer:""")


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


def get_retriever(source_filter=None):
    vectorstore = _build_vectorstore()
    search_kwargs = {
        "k": config.TOP_K,
        "fetch_k": config.MMR_FETCH_K,
        "lambda_mult": config.MMR_LAMBDA,
    }
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}
    return vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] ({src} p.{page})\n{d.page_content}")
    return "\n\n".join(parts)


def ask(question, llm=None):
    source_filter = detect_source_filter(question)
    retriever = get_retriever(source_filter=source_filter)
    llm = llm or ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    docs = retriever.invoke(question)
    context = format_context(docs)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "sources": docs, "filtered_to": source_filter}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = ask(q)
    if result["filtered_to"]:
        print(f"[filtered to: {result['filtered_to']}]")
    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources ===")
    for i, d in enumerate(result["sources"], 1):
        print(f"  [{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}")
