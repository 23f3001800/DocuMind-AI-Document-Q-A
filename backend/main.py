import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
import json
from pathlib import Path

from config import settings
from schemas import (
    IngestResponse, QueryRequest, QueryResponse,
    Citation, EvalQuestion, EvalReport
)
from ingestion import load_pdf, chunk_text
from vectorstore import add_chunks, list_collections
from retriever import hybrid_retrieve
from generator import generate_answer
from evaluator import evaluate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models at startup so first request is fast
    print("Loading embedding and reranker models...")
    from embeddings import get_embedding_model, get_reranker_model
    get_embedding_model()
    get_reranker_model()
    print("Models ready.")
    yield
    print("Shutting down DocuMind RAG.")


app = FastAPI(
    title="DocuMind RAG API",
    description="Production-style document Q&A with hybrid retrieval, reranking, and evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "documind-rag",
        "collections": list_collections(),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    collection: str = "default",
):
    """Upload a PDF, chunk it, embed it, and store in ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # Parse
    pages = load_pdf(file_bytes, file.filename)
    if not pages:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    # Chunk
    chunks = chunk_text(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Embed and store
    n = add_chunks(chunks, collection_name=collection)

    return IngestResponse(
        message="PDF ingested successfully.",
        filename=file.filename,
        chunks_created=n,
        collection=collection,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Retrieve and generate answer with citations."""
    start = time.perf_counter()

    # Hybrid retrieval (BM25 + dense + reranking)
    chunks = hybrid_retrieve(
        query=request.question,
        collection_name=request.collection,
        top_k_final=request.top_k,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found. Please upload a PDF first."
        )

    # Generate answer
    answer = generate_answer(request.question, chunks)

    # Build citations
    citations = [
        Citation(
            source=c["source"],
            page=c["page"],
            chunk_id=f"chunk_{i}",
            content_preview=c["text"][:120] + "...",
        )
        for i, c in enumerate(chunks)
    ]

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return QueryResponse(
        question=request.question,
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
    )


@app.post("/evaluate", response_model=EvalReport)
async def run_evaluation(questions: List[EvalQuestion]):
    """Run RAG evaluation on a set of Q&A pairs."""
    if not questions:
        raise HTTPException(status_code=400, detail="No questions provided.")
    if len(questions) > 50:
        raise HTTPException(status_code=400, detail="Max 50 questions per eval run.")

    report = evaluate(questions)

    # Persist eval results
    Path("../eval").mkdir(exist_ok=True)
    with open("../eval/eval_results.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    return report


@app.get("/collections")
async def collections():
    return {"collections": list_collections()}


@app.post("/debug-pdf")
async def debug_pdf(file: UploadFile = File(...)):
    """Temporary debug endpoint — remove after testing."""
    import fitz
    file_bytes = await file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    result = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        result.append({
            "page": i + 1,
            "char_count": len(text),
            "text_preview": text[:200] if text else "NO TEXT FOUND",
        })
    doc.close()
    return {
        "filename": file.filename,
        "total_pages": len(result),
        "pages": result,
    }