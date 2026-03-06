import html
import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
from pathlib import Path

# ── Direct service imports (no FastAPI middleman) ────────────────────────────
from src.config.pinecone_dp import pinecone_connection
from src.services.rag_service import get_rag_response, rag_pipeline
from src.core.constants import FALLBACK_MESSAGE

ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assist",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Global */
        .block-container { max-width: 820px; padding-top: 2rem; }

        /* Header */
        .app-header {
            text-align: center;
            padding: 1.5rem 0 0.5rem;
        }
        .app-header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .app-header p {
            font-size: 1rem;
            opacity: 0.7;
            margin-top: 0;
        }

        /* Divider */
        .section-divider {
            border: none;
            border-top: 1px solid rgba(128,128,128,0.25);
            margin: 1.5rem 0;
        }

        /* Cards */
        .result-card {
            background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-top: 1rem;
            border-left: 4px solid #4f8bf9;
            color: #1a1a2e !important;
        }
        .result-card .answer-text {
            font-size: 1.05rem;
            line-height: 1.75;
            color: #1a1a2e;
        }
        .result-card .source-link {
            display: inline-block;
            margin-top: 0.75rem;
            font-size: 0.9rem;
            color: #4f8bf9;
        }

        .upload-result-card {
            background: linear-gradient(135deg, #f0faf4 0%, #e4f5ec 100%);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-top: 1rem;
            border-left: 4px solid #2ecc71;
            color: #1a1a2e !important;
        }

        .error-card {
            background: linear-gradient(135deg, #fef4f4 0%, #fde8e8 100%);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            border-left: 4px solid #e74c3c;
            color: #7a1a1a;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        .sidebar-section-title {
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.6;
            margin-bottom: 0.25rem;
        }

        /* Chat history items */
        .history-item {
            padding: 0.6rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.35rem;
            font-size: 0.88rem;
            cursor: default;
            transition: background 0.15s;
        }
        .history-item:hover { background: rgba(79,139,249,0.08); }
            .history-q { font-weight: 600; color: #eaf6ff; }
        .history-a { opacity: 0.72; margin-top: 2px; }

        /* Hide Streamlit branding */
        #MainMenu, footer, header { }
            /* Increase tab size */
            .stTabs { max-width: 1500px !important; width: 100% !important; }
            .stTabs .css-1wmy9hl { min-height: 600px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ── Helper Functions ─────────────────────────────────────────────────────────
def extract_pages_from_file(file_path: str, filename: str) -> list:
    """Extract text pages from PDF, TXT, or DOCX files."""
    ext = Path(filename).suffix.lower()
    pages = []

    if ext == ".pdf":
        with fitz.open(file_path) as doc:
            for page_idx, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "page_content": text,
                        "filename": filename,
                        "page_number": page_idx + 1,
                        "file_path": file_path,
                    })

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            pages.append({
                "page_content": text,
                "filename": filename,
                "page_number": 1,
                "file_path": file_path,
            })

    elif ext == ".docx":
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if text:
                pages.append({
                    "page_content": text,
                    "filename": filename,
                    "page_number": 1,
                    "file_path": file_path,
                })
        except ImportError:
            st.warning(f"python-docx is not installed; skipped {filename}.")

    return pages


def process_uploaded_files(uploaded_files) -> dict:
    """
    Directly process Streamlit-uploaded files through the RAG pipeline:
    extract text → split → embed → upsert to Pinecone.
    """
    try:
        pinecone_connection()

        all_pages = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_files:
                file_path = os.path.join(tmpdir, f.name)
                with open(file_path, "wb") as fp:
                    fp.write(f.getvalue())
                pages = extract_pages_from_file(file_path, f.name)
                all_pages.extend(pages)

            if not all_pages:
                return {"error": "No readable content found in the uploaded files."}

            chunks = rag_pipeline.split_chunks(all_pages)
            embeddings = rag_pipeline.create_embeddings(chunks)
            vectors_stored = rag_pipeline.add_embeddings_to_pinecone(embeddings)

        return {
            "documents_processed": len(all_pages),
            "vectors_stored": vectors_stored,
        }

    except Exception as exc:
        return {"error": str(exc)}


def query_rag(query: str, top_k: int, min_score: float) -> dict:
    """Directly call the RAG service — no HTTP request needed."""
    response = get_rag_response(query=query, top_k=top_k, min_score=min_score)
    if hasattr(response, "dict"):
        return response.dict()
    return {"answer": str(response)}


def truncate(text: str, length: int = 80) -> str:
    return text[:length] + "…" if len(text) > length else text


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️  Settings")

    top_k = st.slider(
        "Top K results",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of document chunks to retrieve.",
    )
    min_score = st.slider(
        "Min relevance score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum similarity score to include a chunk.",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Chat History ─────────────────────────────────────────────────────────
    st.markdown("### 💬  Chat History")

    if st.session_state.chat_history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        for idx, item in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(
                f"""
                <div class='history-item'>
                    <div class='history-q'>Q: {truncate(item["query"], 60)}</div>
                    <div class='history-a'>• {truncate(item["answer"], 60)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No queries yet. Ask a question to get started.")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='app-header'>
        <h1>🔍 RAG Assist</h1>
        <p>AI-Powered Research Assistant <br> Powered by Groq • Gemini LLMs</p>
        
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_query = st.tabs(["📂  Upload Documents", "💡  Ask a Question"])

# ── Upload Tab ───────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("##### Upload your documents for processing")
    st.caption("Supported formats: PDF, TXT, DOCX")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)}** file(s) selected:")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.text(f"  • {f.name}  ({size_kb:.1f} KB)")

    col_l, col_r = st.columns([3, 1])
    with col_r:
        submit_upload = st.button(
            "Upload & Process",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

    if submit_upload and uploaded_files:
        with st.spinner("Processing documents — this may take a moment…"):
            result = process_uploaded_files(uploaded_files)
            st.session_state.upload_result = result
            if "error" not in result:
                st.session_state.uploader_key += 1
                st.rerun()

    # Display upload result
    if st.session_state.upload_result:
        res = st.session_state.upload_result
        if "error" in res:
            st.markdown(
                f"<div class='error-card'>⚠️ {res['error']}</div>",
                unsafe_allow_html=True,
            )
        else:
            docs = res.get("documents_processed", 0)
            vecs = res.get("vectors_stored", 0)
            st.markdown(
                f"""
                <div class='upload-result-card'>
                    <strong>✅  Upload Successful</strong><br>
                    <span style='font-size:0.95rem'>
                        Documents processed: <strong>{docs}</strong><br>
                        Vectors stored: <strong>{vecs}</strong>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Query Tab ────────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("##### Ask a question about your documents")

    query_input = st.text_area(
        "Your question",
        placeholder="e.g. What services does the company provide?",
        height=100,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        submit_query = st.button(
            "Submit",
            type="primary",
            use_container_width=True,
            disabled=not query_input.strip(),
        )

    if submit_query and query_input.strip():
        with st.spinner("Searching documents and generating answer…"):
            try:
                result = query_rag(query_input.strip(), top_k, min_score)

                answer_text = result.get("answer", "No answer returned.")
                source_url = None

                # Separate source URL if appended to the answer
                if "\nSource: http" in answer_text:
                    parts = answer_text.rsplit("\nSource: ", 1)
                    answer_text = parts[0]
                    source_url = parts[1].strip()

                answer_text = html.escape(answer_text).replace("\n", "<br>")

                # Build card HTML
                source_html = ""
                if source_url:
                    source_html = (
                        f"<a class='source-link' href='{source_url}' target='_blank'>"
                        f"📎 Source: {truncate(source_url, 70)}</a>"
                    )

                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div class='answer-text'>{answer_text}</div>
                        {source_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Save to history
                st.session_state.chat_history.append(
                    {"query": query_input.strip(), "answer": answer_text}
                )

            except Exception as exc:
                st.markdown(
                    f"<div class='error-card'>⚠️ {exc}</div>",
                    unsafe_allow_html=True,
                )
