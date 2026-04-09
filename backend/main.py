from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="DocuMind RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "service": "documind-rag"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    return {
        "message": "PDF received successfully.",
        "filename": file.filename
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    return QueryResponse(
        answer="This is a placeholder RAG answer.",
        citations=[Citation(source="sample.pdf", page=1, chunk_id=0)],
        latency_ms=12.5
    )