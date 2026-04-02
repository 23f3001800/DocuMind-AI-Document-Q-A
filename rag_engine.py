# rag_engine.py
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate

# ── Models (loaded once, reused) ────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )

# ── Ingest a PDF ────────────────────────────────────────────
def ingest_pdf(uploaded_file):
    """Takes a Streamlit uploaded file, returns a retriever."""
    # Save to temp file (Streamlit gives bytes, PyPDFLoader needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    # Load + split
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    # Add metadata — which page each chunk came from
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # Embed + store (in-memory for demo, persistent for production)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # MMR retriever — diverse, relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    return retriever, len(chunks)

# ── Build conversational RAG chain ──────────────────────────
def build_chain(retriever):
    llm = get_llm()

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain

# ── Ask a question ───────────────────────────────────────────
def ask(chain, question):
    result = chain.invoke({"question": question})
    answer = result["answer"]
    sources = result["source_documents"]

    # Format source citations
    citations = []
    seen = set()
    for doc in sources:
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:120].strip().replace("\n", " ")
        key = f"p{page}"
        if key not in seen:
            citations.append({"page": page + 1, "snippet": snippet})
            seen.add(key)

    return answer, citations