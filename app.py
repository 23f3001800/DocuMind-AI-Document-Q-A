# app.py
import streamlit as st
from dotenv import load_dotenv
from rag_engine import ingest_pdf, build_chain, retrieve, stream_tokens

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

    # ── User bubble ──────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # ── Assistant bubble ─────────────────────────────────────
    with st.chat_message("assistant", avatar="🧠"):

        # Step 1: Retrieve context (spinner during this ~1-2s step)
        with st.spinner("🔍 Searching document…"):
            try:
                context, citations = retrieve(st.session_state.chain, question)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.stop()

        # Step 2: Stream answer using Streamlit's native write_stream()
        # st.write_stream() accepts a generator of strings, renders
        # each token immediately, and returns the full joined string.
        try:
            full_text = st.write_stream(
                stream_tokens(st.session_state.chain, context, question)
            )
        except Exception:
            # Fallback: non-streaming if streaming fails on this env
            full_text = chain_dict["llm_chain"].invoke(
                {"context": context, "input": question}
            )
            st.markdown(full_text)

        # Step 3: Citations (expanded so user sees them immediately)
        if citations:
            with st.expander(f"📚 {len(citations)} source(s)", expanded=True):
                for c in citations:
                    st.markdown(f"""
                    <div class="cite-card">
                        <div class="cite-page">📄 Page {c['page']}</div>
                        <div class="cite-snippet">"{c['snippet']}…"</div>
                    </div>""", unsafe_allow_html=True)

    # ── Save to history ───────────────────────────────────────
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   full_text,
        "citations": citations,
    })
