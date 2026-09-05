from pathlib import Path
import os
from dotenv import load_dotenv

import ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

load_dotenv()

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("QDRANT_VECTOR_DIM", "768"))

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(
        file=Path(path)
    )

    texts = [
        doc.text
        for doc in docs
        if getattr(doc, "text", None)
    ]

    chunks = []

    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts
    )

    return response["embeddings"]