
# RAG Document Q&A

A production-grade Retrieval-Augmented Generation (RAG) system
for querying any PDF document using semantic search.

**Live demo:** https://23f3001800-documind-ai-document-q-a-app-6pcxhe.streamlit.app

## Architecture
PDF Upload → Recursive Chunking → FAISS Dense Retrieval
→ Cross-Encoder Reranking → Gemini 2.5 Flash → Streamed Answer

## Tech Stack
- **Retrieval:** FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **Reranking:** Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **LLM:** Gemini 2.5 Flash via LangChain
- **Backend:** FastAPI (async) + Pydantic v2
- **Frontend:** Streamlit
- **Evaluation:** Custom Precision@K, Recall@K, MRR, Hit Rate@K

## Evaluation Results (71-page document, 5 questions)
| Config | Precision@4 | Recall@4 | MRR | Hit Rate@4 |
|--------|-------------|----------|-----|------------|
| chunk=256 | 0.20 | 0.70 | 0.60 | 0.60 |
| chunk=512 | 0.20 | 0.70 | 0.67 | 0.80 |
| chunk=1024 | - | - | - | - |

**Selected: chunk_size=512, overlap=50** (best MRR + Hit Rate)



## Tested successfully

![alt text](image.png)
<<<<<<< HEAD
=======


## Run Locally
\`\`\`bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn main:app --reload &
streamlit run app.py
\`\`\`


# DocuMind RAG

Production-style document question answering system with FastAPI, Streamlit, vector retrieval, and citation-grounded responses.

## Why I built this
This project demonstrates AI engineering skills required for production document intelligence systems:
- document ingestion
- embeddings and vector search
- retrieval-augmented generation
- API-first deployment
- evaluation and observability readiness

## Features
- PDF upload and ingestion
- Question answering with citations
- FastAPI backend
- Streamlit frontend
- Evaluation-ready architecture

## Tech Stack
- FastAPI
- Streamlit
- LangChain
- ChromaDB / FAISS
- Sentence Transformers
- RAGAS

## Architecture
Upload PDF → chunk → embed → retrieve → generate answer → return citations

## Status
MVP scaffold complete. Retrieval and evaluation integration in progress.
>>>>>>> 806ac73 ( re-structuring the files and adding the fast api endpoints)



