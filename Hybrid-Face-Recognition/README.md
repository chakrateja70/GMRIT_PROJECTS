# 👤 Hybrid Face Recognition Pipeline

A high-performance, lightweight hybrid face recognition system. This pipeline integrates **RetinaFace** for robust detection, **FaceNet** for deep feature extraction, and **Pinecone** for scalable vector similarity search.

Powered by a **FastAPI** backend and a responsive **Single-Page Application (SPA)** frontend with real-time progress streaming via Server-Sent Events (SSE).

---

## 🏗️ System Architecture
`Video/Image Source` → `RetinaFace (Detection)` → `FaceNet (Embeddings)` → `Pinecone (Vector DB)` → `FastAPI (API/UI)`

### Key Features
* **Hybrid Detection:** Combines state-of-the-art RetinaFace detection with high-speed embedding generation.
* **Vector Database Integration:** Utilizes Pinecone for sub-second similarity searches across millions of faces.
* **Namespace Isolation:** Automatically organizes data by video filename/ID for efficient multi-video searching.
* **Real-time Feedback:** Background processing with **Server-Sent Events (SSE)** for live logs in the Web UI.
* **Developer Friendly:** Dual-entry points via a robust **CLI** or a modern **Web Interface**.

---

## 📂 Repository Layout
* `main.py`: CLI entrypoint for batch processing and quick experiments.
* `run.py`: One-click launcher for the FastAPI server and UI.
* `server.py`: FastAPI backend exposing RESTful endpoints and background job management.
* `store_modes.py` / `search_modes.py`: Core logic for indexing and querying.
* `models.py`: Singleton initialization for FaceNet and Pinecone client.
* `utils.py`: Logic for frame-skipping, batching, and face quality filtering.
* `config.py`: Central configuration for hardware and detection thresholds.
* `frontend/`: Single-page app components (`index.html`, `app.js`, `style.css`).

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.9+**
* **CUDA-enabled GPU** (Optional, but highly recommended for indexing speed).

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/hybrid-face-recognition.git
cd hybrid-face-recognition

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Quickstart — CLI

Store faces from a video (creates a namespace derived from the filename):

```bash
python main.py --mode store --video path/to/video.mp4
```

Search for a person in the stored namespace using a reference image:

```bash
python main.py --mode search --image path/to/person.jpg
```

Bulk store multiple videos:

```bash
python main.py --mode bulk_store --video vid1.mp4 vid2.mp4
```

Batch search (many people, one namespace):

```bash
python main.py --mode batch_search --image a.jpg b.jpg c.jpg
```

---

### Quickstart — Web UI

Start server and open the frontend automatically:

```bash
python run.py         # opens http://localhost:8000 by default
```

Or run with uvicorn directly (dev reload):

```bash
uvicorn server:app --reload --port 8000
```

---

## 🧭 API Overview
* `POST /api/store` — upload a video to index faces (background job with SSE).
* `POST /api/search` — upload image and run instant search against a namespace.
* `POST /api/batch-search` — upload multiple images to search in a namespace.
* `GET /api/status` — returns device, namespaces and total vectors.

---

## ⚙️ Configuration & Environment
* Environment variables
  - `PINECONE_API_KEY` : API key for Pinecone (if using Pinecone).
  - `PINECONE_INDEXNAME` : Target Pinecone index name.

* Configuration
  - Tweak defaults in [config.py](config.py#L1). Common settings: `BASE_FRAME_SKIP`, `MIN_FACE_SIZE`, `MAX_FACES_TO_COLLECT`, `GPU_BATCH_SIZE`, `DIST_THRESHOLD`.

---

## 🛠️ Notes & Troubleshooting
* Ensure `PINECONE_API_KEY` and `PINECONE_INDEXNAME` are set when using Pinecone.
* If TensorFlow or PyTorch on Windows misreport `__version__`, the code includes shims to mitigate that.
* For faster indexing reduce `BASE_FRAME_SKIP` (process fewer frames) or increase `GPU_BATCH_SIZE` when GPU memory allows.

---
