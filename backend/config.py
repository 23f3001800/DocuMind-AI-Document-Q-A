from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    gemini_api_key: str
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_path: str = "./chroma_db"
    top_k_retrieve: int = 10
    top_k_rerank: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 64

    class Config:
        env_file = ".env"


settings = Settings()