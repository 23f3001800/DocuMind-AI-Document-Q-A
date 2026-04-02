# app.py
import streamlit as st
from dotenv import load_dotenv
from rag_engine import ingest_pdf, build_chain, ask

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind — AI Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 DocuMind")
st.caption("Upload a PDF. Ask anything. Get cited answers.")

# ── Session state ─────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Sidebar: upload ──────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded and uploaded.name != st.session_state.doc_name:
        with st.spinner("Reading and indexing your document..."):
            retriever, n_chunks = ingest_pdf(uploaded)
            st.session_state.chain = build_chain(retriever)
            st.session_state.messages = []
            st.session_state.doc_name = uploaded.name
        st.success(f"✓ Indexed {n_chunks} chunks")
        st.info("Start asking questions below!")

    if st.session_state.doc_name:
        st.markdown(f"**Active:** {st.session_state.doc_name}")
        if st.button("Clear & upload new"):
            st.session_state.chain = None
            st.session_state.messages = []
            st.session_state.doc_name = None
            st.rerun()

# ── Chat area ────────────────────────────────────────────────
if not st.session_state.chain:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"**Page {c['page']}:** _{c['snippet']}..._")

# Input
question = st.chat_input("Ask a question about your document...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = ask(st.session_state.chain, question)
        st.write(answer)
        if citations:
            with st.expander("📚 Sources"):
                for c in citations:
                    st.markdown(f"**Page {c['page']}:** _{c['snippet']}..._")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })