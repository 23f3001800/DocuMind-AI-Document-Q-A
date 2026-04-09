from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from embeddings import embed_query, rerank
from vectorstore import query_dense
from config import settings


def bm25_search(query: str, corpus: List[Dict], top_k: int = 10) -> List[Tuple[int, float]]:
    """BM25 keyword retrieval over corpus."""
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def reciprocal_rank_fusion(
    dense_hits: List[Dict],
    bm25_indices: List[Tuple[int, float]],
    all_docs: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """Combine dense and BM25 scores using RRF formula: 1 / (k + rank)."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    # Dense ranking
    for rank, hit in enumerate(dense_hits):
        uid = f"{hit['source']}_p{hit['page']}_{hit['text'][:20]}"
        rrf_scores[uid] = rrf_scores.get(uid, 0) + 1 / (k + rank + 1)
        doc_map[uid] = hit

    # BM25 ranking
    for rank, (idx, _) in enumerate(bm25_indices):
        doc = all_docs[idx]
        uid = f"{doc['source']}_p{doc['page']}_{doc['text'][:20]}"
        rrf_scores[uid] = rrf_scores.get(uid, 0) + 1 / (k + rank + 1)
        if uid not in doc_map:
            doc_map[uid] = doc

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[uid] for uid in sorted_ids]


def hybrid_retrieve(
    query: str,
    collection_name: str = "default",
    top_k_retrieve: int = None,
    top_k_final: int = None,
) -> List[Dict]:
    """
    Full pipeline:
    1. Dense retrieval from ChromaDB
    2. BM25 on the dense hits
    3. RRF fusion
    4. Cross-encoder reranking
    """
    top_k_retrieve = top_k_retrieve or settings.top_k_retrieve
    top_k_final = top_k_final or settings.top_k_rerank

    # Step 1: Dense retrieval
    query_emb = embed_query(query)
    dense_hits = query_dense(query_emb, collection_name, top_k=top_k_retrieve)

    if not dense_hits:
        return []

    # Step 2: BM25 on the same candidate pool
    bm25_indices = bm25_search(query, dense_hits, top_k=top_k_retrieve)

    # Step 3: RRF fusion
    fused = reciprocal_rank_fusion(dense_hits, bm25_indices, dense_hits)

    # Step 4: Cross-encoder reranking
    if len(fused) > 1:
        passages = [doc["text"] for doc in fused]
        top_indices = rerank(query, passages, top_k=top_k_final)
        return [fused[i] for i in top_indices]

    return fused[:top_k_final]