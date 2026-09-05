import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
import requests
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")
if not FASTAPI_URL:
    raise ValueError("FASTAPI_URL is not set in the environment")

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered"
)

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False
    )

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())

    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


def run_async(coro):
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


st.title("📄 Upload a PDF to Ingest")

uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded is not None:

    if st.button("Ingest PDF"):

        with st.spinner("Uploading and triggering ingestion..."):

            path = save_uploaded_pdf(uploaded)

            run_async(
                send_rag_ingest_event(path)
            )

            time.sleep(0.3)

        st.success(
            f"Triggered ingestion for: {path.name}"
        )


st.divider()

st.title("💬 Ask a Question About Your PDFs")

with st.form("rag_query_form"):

    question = st.text_input(
        "Your question"
    )

    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    submitted = st.form_submit_button("Ask")


if submitted and question.strip():

    try:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            response = requests.post(
                f"{FASTAPI_URL}/query",
                json={
                    "question": question.strip(),
                    "top_k": int(top_k)
                },
                timeout=120
            )

            response.raise_for_status()

            output = response.json()

            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")

        st.write(
            answer or "(No answer generated)"
        )

        if sources:

            st.caption("Sources")

            for source in sources:
                st.write(f"- {source}")

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI server. "
        )

    except requests.exceptions.HTTPError as e:

        st.error(
            f"API error: {e}"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )