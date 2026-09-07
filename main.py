import asyncio
import logging
import uuid
import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from pydantic import BaseModel
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import requests

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_MODEL = os.getenv(
    "OPENROUTER_CHAT_MODEL",
    "openrouter/free"
)

def generate_answer(user_content: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": OPENROUTER_CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only "
                        "the provided context. Never use "
                        "outside knowledge."
                    )
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 0.2
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

SESSION_EXPIRY_MINUTES = int(
    os.getenv("SESSION_EXPIRY_MINUTES", "5")
)

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

def cleanup_expired_data():
    try:
        QdrantStorage().delete_expired_sessions()
    except Exception as e:
        print(f"Cleanup warning: {e}")

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id
        )

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        
        session_id = ctx.event.data.get("session_id")

        cleanup_expired_data()

        vecs = embed_texts(chunks)

        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{i}"
                )
            )
            for i in range(len(chunks))
        ]

        expires_at = (
            datetime.now(timezone.utc).timestamp()
            + SESSION_EXPIRY_MINUTES * 60
        )

        payloads = [
            {
                "source": source_id,
                "session_id": session_id,
                "text": chunks[i],
                "expires_at": expires_at
            }
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(
            ids,
            vecs,
            payloads
        )

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    chunk_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunk_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(
        question: str,
        top_k: int = 5,
        source_ids: list[str] | None = None
    ) -> RAGSearchResult:

        query_vec = embed_texts([question])[0]

        found = QdrantStorage().search(
            query_vector=query_vec,
            top_k=top_k,
            source_ids=source_ids
        )

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )

    question = ctx.event.data["question"]

    top_k = ctx.event.data.get(
        "top_k",
        5
    )

    source_ids = ctx.event.data.get(
        "source_ids",
        []
    )

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(
            question,
            top_k,
            source_ids
        ),
        output_type=RAGSearchResult
    )

    if not found.contexts:
        return {
            "answer": "I could not find relevant information in the uploaded PDF.",
            "sources": [],
            "num_contexts": 0
        }

    context_block = "\n\n".join(
        f"- {context}"
        for context in found.contexts
    )

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using only the provided context."
    )

    def _generate_answer():
        return generate_answer(user_content)

    answer = await ctx.step.run(
        "generate-answer",
        _generate_answer
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


app = FastAPI()


async def periodic_cleanup():
    while True:
        cleanup_expired_data()
        await asyncio.sleep(60)


@app.on_event("startup")
async def start_periodic_cleanup():
    asyncio.create_task(periodic_cleanup())


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source_ids: list[str] | None = None
    session_id: str | None = None


class CleanupRequest(BaseModel):
    session_id: str


class HeartbeatRequest(BaseModel):
    source_ids: list[str]
    session_id: str | None = None


class IngestRequest(BaseModel):
    pdf_path: str
    source_id: str
    session_id: str | None = None


@app.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        await inngest_client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={
                    "pdf_path": request.pdf_path,
                    "source_id": request.source_id,
                    "session_id": request.session_id,
                },
            )
        )

        return {"message": "Ingest event queued"}

    except Exception as e:
        logging.getLogger("uvicorn.error").warning(f"Failed to queue ingest event: {e}")
        return {"message": "Failed to queue ingest event (see server logs)"}
@app.post("/heartbeat")
async def heartbeat(request: HeartbeatRequest):

    if not request.source_ids or not request.session_id:
        return {"message": "No active session"}

    cleanup_expired_data()

    expires_at = (
        datetime.now(timezone.utc).timestamp()
        + SESSION_EXPIRY_MINUTES * 60
    )

    QdrantStorage().refresh_session(
        request.session_id,
        expires_at
    )

    return {"message": "Session refreshed"}


@app.post("/query")
async def query_pdf(request: QueryRequest):

    cleanup_expired_data()

    if not request.source_ids:

        return {
            "answer": "No PDFs are currently loaded in this session.",
            "sources": [],
            "num_contexts": 0
        }

    if request.session_id:

        expires_at = (
            datetime.now(timezone.utc).timestamp()
            + SESSION_EXPIRY_MINUTES * 60
        )

        QdrantStorage().refresh_session(
            request.session_id,
            expires_at
        )

    query_vec = embed_texts(
        [request.question]
    )[0]

    found = QdrantStorage().search(
        query_vector=query_vec,
        top_k=request.top_k,
        source_ids=request.source_ids
    )

    if not found["contexts"]:

        return {
            "answer": (
                "I could not find relevant information "
                "in the uploaded PDF."
            ),
            "sources": [],
            "num_contexts": 0
        }

    context_block = "\n\n".join(
        f"- {context}"
        for context in found["contexts"]
    )

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {request.question}\n\n"
        "Answer concisely using only the provided context. "
        "Do not use any knowledge outside the provided context."
    )

    answer = generate_answer(user_content)

    return {
        "answer": answer,
        "sources": found["sources"],
        "num_contexts": len(found["contexts"])
    }


@app.delete("/cleanup")
async def cleanup_session(request: CleanupRequest):

    if not request.session_id:
        return {"message": "No active session"}

    try:
        QdrantStorage().delete_session(
            request.session_id
        )
        return {"message": "Session data deleted"}
    except Exception as e:
        # Log and return a safe response so UI doesn't error out
        logging.getLogger("uvicorn.error").warning(f"Failed to delete session: {e}")
        return {"message": "Failed to delete session (see server logs)"}


@app.delete("/cleanup-all")
async def cleanup_all_sessions():

    try:
        QdrantStorage().delete_all()
        return {"message": "All session data deleted"}
    except Exception as e:
        logging.getLogger("uvicorn.error").warning(f"Failed to delete all sessions: {e}")
        return {"message": "Failed to delete all sessions (see server logs)"}


inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[
        rag_ingest_pdf,
        rag_query_pdf_ai
    ]
)
