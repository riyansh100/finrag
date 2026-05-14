"""Streamlit UI for FinRAG."""

import streamlit as st
from langchain_ollama import ChatOllama

import config
from query import get_retriever, format_context, PROMPT
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="FinRAG", page_icon="📄", layout="wide")
st.title("FinRAG")
st.caption(f"Local RAG over your PDFs · {config.LLM_MODEL} · top-{config.TOP_K}")


@st.cache_resource(show_spinner="Loading retriever and LLM...")
def load_chain():
    retriever = get_retriever()
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    chain = PROMPT | llm | StrOutputParser()
    return retriever, chain


retriever, chain = load_chain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for i, d in enumerate(turn["sources"], 1):
                    st.markdown(
                        f"**[{i}] {d.metadata.get('source')} — p.{d.metadata.get('page')}**"
                    )
                    st.text(d.page_content)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            docs = retriever.invoke(question)
            context = format_context(docs)
            answer = chain.invoke({"context": context, "question": question})
        st.markdown(answer)
        with st.expander("Sources"):
            for i, d in enumerate(docs, 1):
                st.markdown(
                    f"**[{i}] {d.metadata.get('source')} — p.{d.metadata.get('page')}**"
                )
                st.text(d.page_content)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": docs}
    )
