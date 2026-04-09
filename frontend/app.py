import streamlit as st
import requests
import json

st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
    collection = st.text_input("Collection Name", value="default")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)

    st.divider()
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.ok:
                st.success(f"API is online ✅\nCollections: {r.json().get('collections')}")
            else:
                st.error(f"API error: {r.text}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

# ── Main ──────────────────────────────────────────────────
st.title("📄 DocuMind RAG")
st.caption("Production-style document Q&A · Hybrid Retrieval · Citation-grounded Answers")

tabs = st.tabs(["📤 Ingest", "💬 Query", "📊 Evaluate"])

# ── Tab 1: Ingest ─────────────────────────────────────────
with tabs[0]:
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        if st.button("📥 Ingest into Vector Store"):
            with st.spinner("Extracting, chunking, embedding..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    r = requests.post(
                        f"{api_url}/ingest?collection={collection}",
                        files=files,
                        timeout=120,
                    )
                    if r.ok:
                        data = r.json()
                        st.success(
                            f"✅ Ingested **{data['filename']}** → "
                            f"**{data['chunks_created']}** chunks into collection `{data['collection']}`"
                        )
                    else:
                        st.error(f"Ingestion failed: {r.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API not reachable: {e}")

# ── Tab 2: Query ──────────────────────────────────────────
with tabs[1]:
    st.subheader("Ask a question")
    question = st.text_area("Your question", height=80)

    if st.button("🚀 Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    r = requests.post(
                        f"{api_url}/query",
                        json={
                            "question": question,
                            "collection": collection,
                            "top_k": top_k,
                        },
                        timeout=60,
                    )
                    if r.ok:
                        data = r.json()
                        st.subheader("Answer")
                        st.write(data["answer"])

                        st.caption(
                            f"Latency: {data['latency_ms']} ms · "
                            f"Method: {data['retrieval_method']}"
                        )

                        with st.expander("📎 Citations"):
                            for c in data["citations"]:
                                st.markdown(
                                    f"**{c['source']}** · Page {c['page']}"
                                )
                                st.caption(c["content_preview"])
                                st.divider()
                    else:
                        st.error(f"Query failed: {r.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API not reachable: {e}")

# ── Tab 3: Evaluate ───────────────────────────────────────
with tabs[2]:
    st.subheader("Evaluation")
    st.info(
        "Upload a JSON file with Q&A pairs to evaluate your RAG pipeline. "
        "Format: list of {question, ground_truth, collection} objects."
    )

    eval_file = st.file_uploader("Upload eval questions (JSON)", type=["json"])

    if eval_file:
        eval_data = json.loads(eval_file.read())
        st.write(f"Loaded **{len(eval_data)}** questions.")

        if st.button("▶️ Run Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    r = requests.post(
                        f"{api_url}/evaluate",
                        json=eval_data,
                        timeout=300,
                    )
                    if r.ok:
                        report = r.json()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Questions", report["total"])
                        col2.metric("Avg ROUGE-L", report["avg_rouge_l"])
                        col3.metric("Avg Faithfulness", report["avg_faithfulness"])

                        with st.expander("📋 Detailed Results"):
                            for res in report["results"]:
                                st.markdown(f"**Q:** {res['question']}")
                                st.markdown(f"**A:** {res['answer']}")
                                st.caption(
                                    f"ROUGE-L: {res['rouge_l']} · "
                                    f"Faithfulness: {res['faithfulness_score']}"
                                )
                                st.divider()
                    else:
                        st.error(f"Evaluation failed: {r.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API not reachable: {e}")