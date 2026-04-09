import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict


def load_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """Load PDF bytes and extract page-level text."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": filename,
            })
    doc.close()
    return pages


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:
    """Split page text into overlapping chunks."""
    chunks = []
    chunk_id = 0

    for page_data in pages:
        text = page_data["text"]
        words = text.split()
        step = max(1, chunk_size - chunk_overlap)

        i = 0
        while i < len(words):
            chunk_words = words[i: i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{chunk_id}",
                    "text": chunk_text,
                    "page": page_data["page"],
                    "source": page_data["source"],
                })
                chunk_id += 1

            i += step

    return chunks