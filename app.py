"""Streamlit UI for FinRAG."""

import streamlit as st
from langchain_ollama import ChatOllama

import config
from query import (
    PROMPT,
    detect_source_filter,
    format_context,
    is_numeric_question,
    retrieve,
)
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="FinRAG", page_icon="📄", layout="wide")
st.title("FinRAG")
st.caption(f"Local RAG over your PDFs · {config.LLM_MODEL} · top-{config.TOP_K}")


@st.cache_resource(show_spinner="Loading LLM...")
def load_chain():
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    return PROMPT | llm | StrOutputParser()


chain = load_chain()

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(docs):
    with st.expander("Sources"):
        for i, d in enumerate(docs, 1):
            kind = d.metadata.get("type", "text")
            tag = {"table": "📊 TABLE", "figure": "🖼️ FIGURE"}.get(kind, "📄 TEXT")
            st.markdown(
                f"**[{i}] {d.metadata.get('source')} — p.{d.metadata.get('page')} · {tag}**"
            )
            if kind == "table":
                st.markdown(d.page_content)
            else:
                st.text(d.page_content)

question = st.chat_input("Ask a question about your documents...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn.get("flags"):
            st.caption(" · ".join(turn["flags"]))
        st.markdown(turn["content"])
        if turn.get("sources"):
            render_sources(turn["sources"])

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            source_filter = detect_source_filter(question)
            docs, numeric = retrieve(question, source_filter=source_filter)
            context = format_context(docs)
            answer = chain.invoke({"context": context, "question": question})

        flags = []
        if source_filter:
            flags.append(f"Filtered to: `{source_filter}`")
        if numeric:
            flags.append("Numeric intent → table-biased retrieval")
        if flags:
            st.caption(" · ".join(flags))
        st.markdown(answer)
        render_sources(docs)

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
        "flags": flags,
    })
