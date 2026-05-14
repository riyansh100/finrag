"""RAG query: retrieve from Chroma, answer with Ollama llama3.1."""

import sys
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config


PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions using only the provided context.
If the answer is not in the context, say you don't know. Be concise.

Context:
{context}

Question: {question}

Answer:""")


def get_retriever():
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.TOP_K,
            "fetch_k": config.MMR_FETCH_K,
            "lambda_mult": config.MMR_LAMBDA,
        },
    )


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] ({src} p.{page})\n{d.page_content}")
    return "\n\n".join(parts)


def ask(question, retriever=None, llm=None):
    retriever = retriever or get_retriever()
    llm = llm or ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    docs = retriever.invoke(question)
    context = format_context(docs)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "sources": docs}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = ask(q)
    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources ===")
    for i, d in enumerate(result["sources"], 1):
        print(f"  [{i}] {d.metadata.get('source')} p.{d.metadata.get('page')}")
