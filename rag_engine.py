# rag_engine.py
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# ── Models (loaded once, reused) ────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("gemini_api_key"),
        temperature=0.7,
        streaming=True
    )


# ── Ingest a PDF ────────────────────────────────────────────
def ingest_pdf(uploaded_file):
    """Takes a Streamlit uploaded file, returns a retriever."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
 # Save to temp file (Streamlit gives bytes, PyPDFLoader needs a path)
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    # Load + split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(pages)

    # metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    return retriever, len(chunks)


# ── Build RAG chain (NEW WAY) ───────────────────────────────
def build_chain(retriever):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.

    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {input}

    Answer clearly and concisely.
    """)

    # Combine docs + LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieval + QA
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain

# ── Non-streaming ask (kept for fallback) ────────────────────
def ask(chain, question):
    result = chain.invoke({"input": question})
    return result["answer"], _extract_citations(result.get("context", []))


# ── Streaming ask ─────────────────────────────────────────────
def ask_stream(chain, question):
    """
    Generator. Yields (token, None) for each token during streaming.
    Yields ("", citations_list) once as the final sentinel when done.

    How chain.stream() chunks arrive:
      chunk 1  →  {"context": [...docs...]}   ← retrieved docs
      chunk 2+ →  {"answer": "Hello"}         ← one token each
    """
    citations = []

    for chunk in chain.stream({"input": question}):

        # First chunk contains the retrieved documents → build citations
        if "context" in chunk:
            citations = _extract_citations(chunk["context"])

        # Subsequent chunks contain one answer token each
        if "answer" in chunk and chunk["answer"]:
            yield chunk["answer"], None   # ← send token to app.py

    # After loop ends → send citations as the final signal
    yield "", citations


# ── Shared helper ─────────────────────────────────────────────
def _extract_citations(docs):
    citations, seen = [], set()
    for doc in docs:
        page = doc.metadata.get("page", 0)
        snippet = doc.page_content[:120].strip().replace("\n", " ")
        key = f"p{page}"
        if key not in seen:
            citations.append({"page": page + 1, "snippet": snippet})
            seen.add(key)
    return citations