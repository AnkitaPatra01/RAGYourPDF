from pathlib import Path
import os
import requests
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL",
    "nvidia/nemotron-3-embed-1b:free"
)

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
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()["data"]

    data.sort(key=lambda item: item["index"])

    return [
        item["embedding"]
        for item in data
    ]