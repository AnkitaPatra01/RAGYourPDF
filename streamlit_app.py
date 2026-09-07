import asyncio
from pathlib import Path
import time
import os
import uuid
import base64

import streamlit as st
import httpx
import requests
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")
if not FASTAPI_URL:
    raise ValueError("FASTAPI_URL is not set in the environment")

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fc;
    }

    h1, h2, h3 {
        color: #1f3b5b;
    }

    .upload-container {
        width: 100%;
        max-width: none;
        margin: 0 auto;
        padding-top: 70px;
    }

    .upload-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 10px;
    }

    .upload-subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 35px;
    }

    .pdf-header {
        font-size: 18px;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 15px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .chat-header {
        font-size: 25px;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 15px;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    .st-key-upload_start_button button {
        min-height: 48px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #d9e2ef;
        border-radius: 12px;
        padding: 22px;
        background-color: #ffffff;
    }

    .st-key-upload_panel {
        background-color: #f3f7fc;
        border: 1px solid #cbdcf0;
        border-radius: 14px;
        padding: 18px;
    }

    div[class*="st-key-left_panel"],
    div[class*="st-key-right_chat_panel"] {
        background-color: #f3f7fc !important;
        border: 1px solid #cbdcf0 !important;
        border-radius: 14px !important;
        padding: 14px 21px !important;
        height: 800px !important;
        min-height: 850px !important;
        max-height: 850px !important;
        box-sizing: border-box !important;
        box-shadow: 2px 2px 2px 2px #9eaca7;
        overflow: hidden !important;
    }

    div[data-testid="stHorizontalBlock"]:has(> div [class*="st-key-left_panel"]),
    div[data-testid="stHorizontalBlock"]:has(> div [class*="st-key-right_chat_panel"]) {
        align-items: stretch;
    }

    div[data-testid="stPopover"] button {
        min-width: 52px !important;
        width: 52px !important;
        height: 48px !important;
        padding: 0 !important;
        background-color: #dceaf8 !important;
        color: #1e3a5f !important;
        border: 1px solid #cbdcf0 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button:focus,
    div[data-testid="stPopover"] button:active {
        background-color: #dceaf8 !important;
        color: #1e3a5f !important;
        border: 1px solid #cbdcf0 !important;
        box-shadow: none !important;
    }

    div[data-testid="stPopover"] button span,
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopover"] button svg {
        color: #1e3a5f !important;
        fill: #1e3a5f !important;
        font-size: 27px !important;
    }

    div[data-testid="stPopover"] button svg:last-child:not(:only-child) {
        display: none !important;
    }

    div[data-testid="stPopover"] button {
        min-width: 52px !important;
        width: 52px !important;
    }

   div[data-testid="stChatInput"] {
    border-radius: 12px !important;
    min-height: 68px !important;
    margin-top: 8px !important;
}

div[data-testid="stChatInput"] textarea {
    min-height: 46px !important;
    font-size: 16px !important;
    padding-top: 12px !important;
    padding-bottom: 12px !important;
}

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        color: #000000 !important;
    }

    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] span {
        color: #000000 !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #8b9099 !important;
        border-radius: 12px !important;
        padding: 8px 12px !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * {
        color: #ffffff !important;
    }

    .assistant-answer {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
        padding: 14px;
        border-radius: 12px;
        line-height: 1.6;
    }

    .assistant-answer * {
        color: #000000 !important;
    }

    div[data-testid="stExpander"] {
        background-color: #f3f4f6 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] details {
        background-color: #f3f4f6 !important;
    }

    div[data-testid="stExpander"] details summary {
        background-color: #d1d5db !important;
        color: #374151 !important;
        border-radius: 0 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stExpander"] details summary:hover {
        background-color: #c4c9d1 !important;
        color: #1f2937 !important;
    }

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary svg {
        color: #374151 !important;
        fill: #374151 !important;
    }

    div[data-testid="stExpander"] p {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:

    response = requests.delete(
        f"{FASTAPI_URL}/cleanup-all",
        timeout=10
    )

    response.raise_for_status()

    st.session_state.clear()

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.active_sources = []
    st.session_state.page = "upload"
    st.session_state.uploaded_pdf_path = None
    st.session_state.uploaded_pdf_name = None
    st.session_state.chat_history = []
    st.session_state.top_k = 5

if "active_sources" not in st.session_state:
    st.session_state.active_sources = []

if "page" not in st.session_state:
    st.session_state.page = "upload"

if "uploaded_pdf_path" not in st.session_state:
    st.session_state.uploaded_pdf_path = None

if "uploaded_pdf_name" not in st.session_state:
    st.session_state.uploaded_pdf_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "top_k" not in st.session_state:
    st.session_state.top_k = 5


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    unique_file_name = f"{uuid.uuid4()}_{file.name}"

    file_path = uploads_dir / unique_file_name
    file_path.write_bytes(file.getbuffer())

    return file_path


async def send_rag_ingest_event(
    pdf_path: Path,
    source_id: str,
    session_id: str
) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{FASTAPI_URL}/ingest",
            json={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": source_id,
                "session_id": session_id,
            },
        )

        resp.raise_for_status()


def run_async(coro):
    return asyncio.run(coro)


def send_heartbeat():

    if not st.session_state.active_sources:
        return

    try:

        requests.post(
            f"{FASTAPI_URL}/heartbeat",
            json={
                "source_ids": st.session_state.active_sources,
                "session_id": st.session_state.session_id
            },
            timeout=10
        )

    except Exception:
        pass


def cleanup_current_session():

    try:

        requests.delete(
            f"{FASTAPI_URL}/cleanup",
            json={
                "session_id": st.session_state.session_id
            },
            timeout=10
        )

    except Exception:
        pass


def display_pdf(pdf_path):

    try:

        pdf_bytes = Path(pdf_path).read_bytes()

        if hasattr(st, "pdf"):

            st.pdf(
                pdf_bytes,
                height=780
            )

        else:

            st.error(
                "Your Streamlit version does not support the native PDF viewer. "
                "Please upgrade Streamlit to use st.pdf."
            )

    except Exception as e:

        st.error(
            f"Unable to display PDF: {str(e)}"
        )


if st.session_state.active_sources:
    send_heartbeat()

if st.session_state.page == "upload":

    st.markdown(
        '<div class="upload-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="upload-title">Chat with your PDF</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="upload-subtitle">'
        'Upload a PDF and start asking questions instantly'
        '</div>',
        unsafe_allow_html=True
    )

    _, upload_col, _ = st.columns([1, 3, 1])

    with upload_col:

        with st.container(key="upload_panel"):

            uploaded = st.file_uploader(
                "Upload your PDF",
                type=["pdf"],
                accept_multiple_files=False
            )

            if uploaded:

                _, button_col, _ = st.columns([1.5, 0.8, 1.5])

                with button_col:

                    if st.button(
                        "Upload and Start Chat",
                        key="upload_start_button",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "Uploading and processing your PDF..."
                        ):

                            if st.session_state.active_sources:
                                cleanup_current_session()

                                st.session_state.session_id = str(uuid.uuid4())
                                st.session_state.active_sources = []
                                st.session_state.chat_history = []

                            path = save_uploaded_pdf(uploaded)

                            source_id = (
                                f"{st.session_state.session_id}:{uploaded.name}"
                            )

                            run_async(
                                send_rag_ingest_event(
                                    path,
                                    source_id,
                                    st.session_state.session_id
                                )
                            )

                            if source_id not in st.session_state.active_sources:

                                st.session_state.active_sources.append(
                                    source_id
                                )

                            st.session_state.uploaded_pdf_path = str(path)

                            st.session_state.uploaded_pdf_name = uploaded.name

                            st.session_state.chat_history = []

                            time.sleep(0.5)

                        st.session_state.page = "chat"

                        st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

elif st.session_state.page == "chat":

    left_col, right_col = st.columns(
        [1, 1],
        gap="large"
    )

    with left_col:
        with st.container(key="left_panel"):
            if st.session_state.uploaded_pdf_name:

                st.markdown(
                    '<div class="pdf-header">📄 ' +
                    st.session_state.uploaded_pdf_name +
                    '</div>',
                    unsafe_allow_html=True
                )

            if st.session_state.uploaded_pdf_path:

                display_pdf(
                    st.session_state.uploaded_pdf_path
                )

            else:

                st.warning(
                    "No PDF available for preview."
                )

    with right_col:

        with st.container(key="right_chat_panel"):
            header_col, settings_col = st.columns(
                [12, 1],
                vertical_alignment="center"
            )

            with header_col:
                st.markdown(
                    '<div class="chat-header">'
                    '💬 Ask questions about your PDF'
                    '</div>',
                    unsafe_allow_html=True
                )

            with settings_col:
                with st.popover(
                    ":material/settings:",
                    help="Retrieval settings"
                ):
                    st.session_state.top_k = st.number_input(
                        "Chunks to retrieve",
                        min_value=1,
                        max_value=20,
                        value=st.session_state.top_k,
                        step=1
                    )

            chat_container = st.container(
                height=650,
                border=True
            )

            with chat_container:

                if not st.session_state.chat_history:

                    st.info(
                        "Ask me anything about the uploaded PDF."
                    )


                for message in st.session_state.chat_history:

                    with st.chat_message(
                        message["role"]
                    ):

                        st.write(
                            message["content"]
                        )

                        if (
                            message["role"] == "assistant"
                            and message.get("sources")
                        ):

                            with st.expander(
                                "Sources",
                                expanded=False
                            ):

                                for source in message["sources"]:

                                    st.write(
                                        f"📄 {source}"
                                    )

            question = st.chat_input(
                "Ask anything about your PDF..."
            )


            if question and question.strip():

                clean_question = question.strip()

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": clean_question
                    }
                )

                with chat_container:

                    with st.chat_message("user"):

                        st.write(clean_question)

                    with st.chat_message("assistant"):

                        try:

                            with st.spinner(
                                "Searching your PDF..."
                            ):

                                response = requests.post(
                                    f"{FASTAPI_URL}/query",
                                    json={
                                        "question": clean_question,
                                        "top_k": int(
                                            st.session_state.top_k
                                        ),
                                        "source_ids": (
                                            st.session_state.active_sources
                                        ),
                                        "session_id": (
                                            st.session_state.session_id
                                        )
                                    },
                                    timeout=120
                                )

                                response.raise_for_status()

                                output = response.json()

                                answer = output.get(
                                    "answer",
                                    ""
                                )

                                sources = output.get(
                                    "sources",
                                    []
                                )

                            st.write(
                                answer or "(No answer generated)"
                            )

                            if sources:

                                with st.expander(
                                    "Sources",
                                    expanded=False
                                ):

                                    for source in sources:

                                        st.write(
                                            f"📄 {source}"
                                        )

                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": (
                                        answer
                                        or "(No answer generated)"
                                    ),
                                    "sources": sources
                                }
                            )

                        except requests.exceptions.ConnectionError:

                            st.error(
                                "Cannot connect to FastAPI server."
                            )

                        except requests.exceptions.Timeout:

                            st.error(
                                "The request took too long. "
                                "Please try again."
                            )

                        except requests.exceptions.HTTPError as e:

                            st.error(
                                f"API error: {e}"
                            )

                        except Exception as e:

                            st.error(
                                f"Error: {str(e)}"
                            )
