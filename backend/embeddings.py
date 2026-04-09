from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
from config import settings

# Load once, reuse across requests
_embedding_model = None
_reranker_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.reranker_model)
    return _reranker_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> List[float]:
    model = get_embedding_model()
    return model.encode([query], show_progress_bar=False)[0].tolist()


def rerank(query: str, passages: List[str], top_k: int = 3) -> List[int]:
    """Return top_k indices ranked by cross-encoder relevance."""
    model = get_reranker_model()
    pairs = [(query, p) for p in passages]
    scores = model.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]