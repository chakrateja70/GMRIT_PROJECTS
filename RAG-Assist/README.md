# RAG-Assist

A **Retrieval-Augmented Generation (RAG)** chatbot built with FastAPI that answers questions from uploaded documents using **Groq (LLaMA 3.1)** and **Google Gemini** LLMs, with **Pinecone** as the vector store.

---

## Features

- Upload PDF and DOCX documents for ingestion
- Semantic search over uploaded documents via Pinecone vector store
- Answers grounded strictly in uploaded document content
- Dual LLM support — Groq (LLaMA 3.1-8b-instant) and Gemini 2.5 Flash
- Streamlit UI for easy interaction
- REST API with Swagger docs at `/docs`

---

## Project Structure

```
RAG-Assist/
├── main.py                        # FastAPI app entry point
├── streamlit_app.py               # Streamlit frontend
├── requirements.txt
└── src/
    ├── routes.py                  # API route definitions (/upload, /query)
    ├── config/
    │   ├── settings.py            # Environment variable loading
    │   └── pinecone_dp.py         # Pinecone client setup
    ├── core/
    │   ├── prompts.py             # LLM prompt templates
    │   ├── constants.py           # App-wide constants
    │   └── exceptions.py          # Custom exceptions
    ├── db/
    │   └── upload.py              # Document ingestion pipeline
    ├── schemas/
    │   └── response.py            # Pydantic request/response models
    ├── services/
    │   ├── rag_service.py         # RAG orchestration logic
    │   ├── llm_service.py         # Groq LLM service
    │   └── llm_service_gemini.py  # Gemini LLM service
    └── utils/
        ├── document_processor.py  # Chunking & embedding pipeline
        └── swagger.py             # Swagger endpoint metadata
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/chakrateja70/GMRIT_PROJECTS.git
cd GMRIT_PROJECTS/RAG-Assist
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the `RAG-Assist/` directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_NAMESPACE=your_namespace
PINECONE_BATCH_SIZE=100
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
POPPLER_PATH=C:/path/to/poppler/bin   # Required for PDF processing on Windows
```

---

## Running the App

### FastAPI Backend

```bash
python main.py
```

API will be available at `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

### Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

---

## API Endpoints

### `POST /upload`
Upload one or more PDF/DOCX files to be processed and stored in Pinecone.

**Request:** `multipart/form-data` with `uploaded_files`

---

### `POST /query`
Query the RAG system with a natural language question.

**Request Body:**
```json
{
  "query": "What is the legal definition of negligence?",
  "top_k": 5,
  "min_score": 0.8
}
```

**Response:**
```json
{
  "statusCode": 200,
  "success": true,
  "query": "...",
  "answer": "...",
  "source_url": "..."
}
```

---

## Tech Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| Backend         | FastAPI, Uvicorn                  |
| LLMs            | Groq (LLaMA 3.1-8b-instant), Gemini 2.5 Flash |
| Vector Store    | Pinecone                          |
| Embeddings      | Sentence Transformers             |
| Document Parse  | PyMuPDF, python-docx, Unstructured |
| Frontend        | Streamlit                         |
| Config          | Pydantic Settings, python-dotenv  |
