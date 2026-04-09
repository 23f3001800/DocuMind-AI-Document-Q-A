from pydantic import BaseModel, Field
from typing import List, Optional


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    collection: str


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    content_preview: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    collection: str = Field(default="default")
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    latency_ms: float
    retrieval_method: str = "hybrid_bm25_dense_reranked"


class EvalQuestion(BaseModel):
    question: str
    ground_truth: str
    collection: str = "default"


class EvalResult(BaseModel):
    question: str
    answer: str
    ground_truth: str
    rouge_l: float
    faithfulness_score: float


class EvalReport(BaseModel):
    total: int
    avg_rouge_l: float
    avg_faithfulness: float
    results: List[EvalResult]