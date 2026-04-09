# # app.py
# import streamlit as st
# from dotenv import load_dotenv
# from rag_engine import ingest_pdf, build_chain, ask_stream

# load_dotenv()

# # ── Page config ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="DocuMind — AI Document Q&A",
#     page_icon="📄",
#     layout="centered"
# )

# st.title("📄 DocuMind")
# st.caption("Upload a PDF. Ask anything. Get cited answers.")

# # ── Session state ─────────────────────────────────────────────
# if "chain" not in st.session_state:
#     st.session_state.chain = None
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "doc_name" not in st.session_state:
#     st.session_state.doc_name = None

# # ── Sidebar: upload ──────────────────────────────────────────
# with st.sidebar:
#     st.header("📁 Document")
#     uploaded = st.file_uploader("Upload a PDF", type="pdf")

#     if uploaded and uploaded.name != st.session_state.doc_name:
#         with st.spinner("Reading and indexing your document..."):
#             retriever, n_chunks = ingest_pdf(uploaded)
#             st.session_state.chain = build_chain(retriever)
#             st.session_state.messages = []
#             st.session_state.doc_name = uploaded.name
#         st.success(f"✓ Indexed {n_chunks} chunks")
#         st.info("Start asking questions below!")

#     if st.session_state.doc_name:
#         st.markdown(f"**Active:** {st.session_state.doc_name}")
#         if st.button("Clear & upload new"):
#             st.session_state.chain = None
#             st.session_state.messages = []
#             st.session_state.doc_name = None
#             st.rerun()

# # ── Chat area ────────────────────────────────────────────────
# if not st.session_state.chain:
#     st.info("👈 Upload a PDF in the sidebar to get started.")
#     st.stop()

# # Render message history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])
#         if msg.get("citations"):
#             with st.expander("📚 Sources"):
#                 for c in msg["citations"]:
#                     st.markdown(f"**Page {c['page']}:** _{c['snippet']}..._")

# # Input
# question = st.chat_input("Ask a question about your document...")

# if question:
#     # ── User bubble (UNCHANGED) ──────────────────────────────
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.write(question)

#     # ── Assistant bubble (REPLACE with streaming) ────────────
#     with st.chat_message("assistant"):

#         # Step A: Show "Thinking…" while waiting for first token.
#         # st.empty() creates a placeholder we can overwrite later.
#         thinking_ph = st.empty()
#         thinking_ph.markdown("⏳ Thinking…")

#         # Step B: Another placeholder for the growing answer text.
#         answer_ph = st.empty()

#         full_text  = ""
#         citations  = []
#         first_token = True

#         # Step C: Loop over the generator from ask_stream().
#         #   - Each iteration gives us one token (or the final sentinel).
#         #   - We update answer_ph.markdown() in-place every token.
#         #   - Appending  ▌  gives a blinking-cursor effect while streaming.
#         for token, done_citations in ask_stream(st.session_state.chain, question):

#             if done_citations is not None:
#                 # This is the final sentinel → grab citations and stop
#                 citations = done_citations
#                 break

#             if first_token:
#                 # First real token arrived → remove "Thinking…"
#                 thinking_ph.empty()
#                 first_token = False

#             # Accumulate text and redraw the same placeholder
#             full_text += token
#             answer_ph.markdown(full_text + " ▌")   # ▌ = live cursor

#         # Step D: Streaming done → remove cursor, show clean final text
#         answer_ph.markdown(full_text)

#         # Step E: Show citations (expanded so user sees them right away)
#         if citations:
#             with st.expander(f"📚 {len(citations)} source(s)", expanded=True):
#                 for c in citations:
#                     st.markdown(f"**Page {c['page']}:** _{c['snippet']}..._")

#     # ── Save to history (UNCHANGED structure) ────────────────
#     st.session_state.messages.append({
#         "role":      "assistant",
#         "content":   full_text,    # ← save final clean text (no cursor)
#         "citations": citations,
#     })

# app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document QA", page_icon="📄")
st.title("📄 RAG Document Q&A")

# ─── Sidebar — Ingest ───────────────────
# Replace the ingest button section in app.py

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload any PDF document to query"
    )

    chunk_size    = st.slider("Chunk Size", 256, 1024, 512, step=128)
    chunk_overlap = st.slider("Chunk Overlap", 25, 200, 50, step=25)
    use_reranker  = st.toggle("Use Reranker", value=True)

    if uploaded_file and st.button("🔄 Ingest Document", type="primary"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            resp = requests.post(
                f"{API_URL}/ingest",
                params={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                },
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ {data['num_chunks']} chunks from {uploaded_file.name}")
                st.session_state["ingested"] = True
                st.session_state["filename"] = uploaded_file.name
            else:
                st.error(f"Error: {resp.json().get('detail', resp.text)}")

    # Health status
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health["document_loaded"]:
            st.success(f"📊 Ready — {health['num_chunks']} chunks")
        else:
            st.warning("⚠️ Upload a PDF to get started")
    except Exception:
        st.error("❌ API not reachable — is `uvicorn main:app` running?")

# ─── Main — Chat ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# In app.py — replace the query block with this

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        citations_placeholder = st.empty()
        full_response = ""

        # Connect to /stream endpoint
        with requests.post(
            f"{API_URL}/stream",
            json={"question": prompt, "top_k": 4, "use_reranker": True},
            stream=True   # ← critical: tells requests not to buffer
        ) as resp:

            for line in resp.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                event = json.loads(line[6:])   # strip "data: "

                if event["type"] == "citations":
                    # Show citations immediately while answer streams
                    with citations_placeholder.expander("📚 Sources"):
                        for i, c in enumerate(event["data"], 1):
                            st.markdown(f"**[{i}] Page {c['page']}**")
                            st.caption(c["snippet"])

                elif event["type"] == "token":
                    full_response += event["data"]
                    # st.write_stream needs a generator — use manual approach
                    citations_placeholder.markdown(full_response + "▌")

                elif event["type"] == "done":
                    citations_placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })