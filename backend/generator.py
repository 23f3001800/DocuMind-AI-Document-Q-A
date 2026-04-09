from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from config import settings


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
    return _llm


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise document assistant. Answer ONLY using the provided context.
If the context does not contain the answer, say: "I could not find this in the uploaded document."

Rules:
- Be factual and concise
- Do not hallucinate or add external knowledge
- Refer to pages when possible

Context:
{context}
""",
    ),
    ("human", "{question}"),
])


def format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Chunk {i+1} | Source: {chunk['source']} | Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant content found in the uploaded document."

    context = format_context(chunks)
    chain = RAG_PROMPT | get_llm() | StrOutputParser()

    return chain.invoke({"context": context, "question": question})