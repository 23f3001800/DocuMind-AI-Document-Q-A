# app.py
import streamlit as st
from dotenv import load_dotenv
from rag_engine import ingest_pdf, build_chain, ask_stream

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
    # ── User bubble (UNCHANGED) ──────────────────────────────
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # ── Assistant bubble (REPLACE with streaming) ────────────
    with st.chat_message("assistant"):

        # Step A: Show "Thinking…" while waiting for first token.
        # st.empty() creates a placeholder we can overwrite later.
        thinking_ph = st.empty()
        thinking_ph.markdown("⏳ Thinking…")

        # Step B: Another placeholder for the growing answer text.
        answer_ph = st.empty()

        full_text  = ""
        citations  = []
        first_token = True

        # Step C: Loop over the generator from ask_stream().
        #   - Each iteration gives us one token (or the final sentinel).
        #   - We update answer_ph.markdown() in-place every token.
        #   - Appending  ▌  gives a blinking-cursor effect while streaming.
        for token, done_citations in ask_stream(st.session_state.chain, question):

            if done_citations is not None:
                # This is the final sentinel → grab citations and stop
                citations = done_citations
                break

            if first_token:
                # First real token arrived → remove "Thinking…"
                thinking_ph.empty()
                first_token = False

            # Accumulate text and redraw the same placeholder
            full_text += token
            answer_ph.markdown(full_text + " ▌")   # ▌ = live cursor

        # Step D: Streaming done → remove cursor, show clean final text
        answer_ph.markdown(full_text)

        # Step E: Show citations (expanded so user sees them right away)
        if citations:
            with st.expander(f"📚 {len(citations)} source(s)", expanded=True):
                for c in citations:
                    st.markdown(f"**Page {c['page']}:** _{c['snippet']}..._")

    # ── Save to history (UNCHANGED structure) ────────────────
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   full_text,    # ← save final clean text (no cursor)
        "citations": citations,
    })
