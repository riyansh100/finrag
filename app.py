"""Streamlit UI for FinRAG."""

import streamlit as st
from langchain_ollama import ChatOllama

import config
from query import ask

st.set_page_config(page_title="FinRAG", page_icon="📄", layout="wide")
st.title("FinRAG")
st.caption(f"Local RAG over your PDFs · {config.LLM_MODEL} · top-{config.TOP_K} · "
           f"memory: last {config.HISTORY_TURNS // 2} Q&A pairs")

@st.cache_resource(show_spinner="Loading LLM...")
def load_llm():
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                      temperature=0)

llm = load_llm()

if "history" not in st.session_state:
    st.session_state.history = []


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.subheader("Chat")
    if st.button("🗑️  New chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    _stored = len(st.session_state.history)
    _sent = min(_stored, config.HISTORY_TURNS)
    st.caption(f"{_stored} message(s) stored · last {_sent} sent to model "
               f"(~{_sent // 2} Q&A pairs)")
    st.divider()
    st.caption("Phase 2 features active:")
    st.markdown(
        "- Hybrid retrieval (BM25 + vector)\n"
        "- Author-filter detection\n"
        "- Numeric-intent → table bias\n"
        "- Figure-intent → figure bias\n"
        "- Conversational memory + query rewriter"
    )


# --- helpers -----------------------------------------------------------------

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


def history_for_llm():
    """Strip UI-only fields from session history before passing to ask()."""
    return [{"role": t["role"], "content": t["content"]}
            for t in st.session_state.history]


# --- replay prior turns ------------------------------------------------------

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn.get("flags"):
            st.caption(" · ".join(turn["flags"]))
        st.markdown(turn["content"])
        if turn.get("sources"):
            render_sources(turn["sources"])


# --- new turn ----------------------------------------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            # Pass history BEFORE appending the new user turn? We just appended it;
            # ask() expects history NOT including the current question, so slice it off.
            prior = history_for_llm()[:-1]
            result = ask(question, history=prior, llm=llm)

        flags = []
        if result.get("rewritten_query"):
            flags.append(f"Rewrote → `{result['rewritten_query']}`")
        if result["filtered_to"]:
            flags.append(f"Filtered to: `{result['filtered_to']}`")
        if result.get("multi_sources"):
            flags.append("Multi-author → per-author retrieval bonus: "
                         + ", ".join(f"`{s}`" for s in result["multi_sources"]))
        if result["numeric"]:
            flags.append("Numeric intent → table-biased retrieval")

        if flags:
            st.caption(" · ".join(flags))
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "flags": flags,
    })
