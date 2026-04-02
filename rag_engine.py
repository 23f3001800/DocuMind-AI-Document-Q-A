# rag_engine.py — rewritten with LCEL (no langchain.chains dependency)

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Embeddings (loaded once) ─────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        streaming=True,
    )

# ── Ingest PDF ────────────────────────────────────────────────
def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    pages = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )
    return retriever, len(chunks)

# ── Build chain (LCEL — no langchain.chains needed) ──────────
def build_chain(retriever):
    """
    Returns a dict with retriever + llm_chain separately.
    We keep them separate so ask_stream() can:
      1. Call retriever first → get docs for citations
      2. Stream llm_chain → token by token output
    """
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based ONLY on the context below.
Be concise. If the answer is not in the context, say so.

Context:
{context}

Question:
{input}

Answer:
""")
    llm_chain = prompt | get_llm() | StrOutputParser()
    return {"retriever": retriever, "llm_chain": llm_chain}

# ── Streaming ask ─────────────────────────────────────────────
def ask_stream(chain_dict, question):
    """
    Generator. Yields (token, None) per token, then ("", citations) at end.
    """
    retriever  = chain_dict["retriever"]
    llm_chain  = chain_dict["llm_chain"]

    # Step 1: retrieve docs synchronously → build citations immediately
    docs = retriever.invoke(question)
    citations = _extract_citations(docs)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 2: stream LLM response token by token
    for token in llm_chain.stream({"context": context, "input": question}):
        yield token, None

    # Step 3: sentinel — signals stream is done, passes citations
    yield "", citations

# ── Non-streaming fallback ────────────────────────────────────
def ask(chain_dict, question):
    retriever = chain_dict["retriever"]
    llm_chain = chain_dict["llm_chain"]
    docs      = retriever.invoke(question)
    context   = "\n\n".join(doc.page_content for doc in docs)
    answer    = llm_chain.invoke({"context": context, "input": question})
    return answer, _extract_citations(docs)

# ── Helper ────────────────────────────────────────────────────
def _extract_citations(docs):
    citations, seen = [], set()
    for doc in docs:
        page    = doc.metadata.get("page", 0)
        snippet = doc.page_content[:120].strip().replace("\n", " ")
        key     = f"p{page}"
        if key not in seen:
            citations.append({"page": page + 1, "snippet": snippet})
            seen.add(key)
    return citations
