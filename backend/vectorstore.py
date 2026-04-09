import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict
from config import settings
from embeddings import embed_texts

# Persistent ChromaDB client
_client = None


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def add_chunks(chunks: List[Dict], collection_name: str = "default") -> int:
    """Embed and store chunks in ChromaDB collection."""
    client = get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"]}
        for c in chunks
    ]
    embeddings = embed_texts(texts)

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i: i + batch_size],
            documents=texts[i: i + batch_size],
            embeddings=embeddings[i: i + batch_size],
            metadatas=metadatas[i: i + batch_size],
        )

    return len(chunks)


def query_dense(
    query_embedding: List[float],
    collection_name: str = "default",
    top_k: int = 10,
) -> List[Dict]:
    """Dense retrieval from ChromaDB."""
    client = get_client()
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"][0]:
        return []

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "score": 1 - dist,  # cosine similarity
        })
    return hits


def list_collections() -> List[str]:
    client = get_client()
    return [c.name for c in client.list_collections()]