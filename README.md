# Agentic Academic RAG Pipeline 🧬📚

An intelligent, self-reflective Retrieval-Augmented Generation (RAG) system for querying and synthesizing academic biomedical literature.

Built on a **LangGraph state machine** — the system doesn't just retrieve and generate, it actively evaluates retrieval quality and reformulates queries if it lacks sufficient evidence before answering.

## 🌟 Key Features

* **Agentic State Machine:** A LangGraph cyclic workflow routes queries, evaluates context sufficiency, and triggers rewrites when retrieval fails.
* **Query Decomposition:** Complex questions are broken down into sub-queries for broader coverage.
* **Hybrid Retrieval:** Combines dense vector search with keyword search, then re-ranks with a **cross-encoder** (ms-marco-MiniLM-L-6-v2).
* **Parent-Child Retrieval:** Child chunk matches automatically fetch the full parent document for richer context.
* **Strict Citation Grounding:** The synthesizer is prompted to use *only* retrieved text and cite every fact `[Document: Title, Year]`.
* **SSE Streaming:** Answers stream token-by-token to the frontend via Server-Sent Events.
* **Semantic Cache:** Repeated similar questions are served instantly from an in-memory cache (cosine similarity ≥ 0.95).
* **Local Privacy-First AI:** Fully local inference powered by **Ollama** (`mistral`, `nomic-embed-text`). No API keys required.
* **Persistent Vector Store:** **ChromaDB** with automatic metadata filtering by year.

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Analyzer │  ─── rejects off-topic queries immediately
└────────┬────────┘
         │ in-scope
         ▼
┌─────────────────┐
│    Retriever    │  ─── dense + keyword search → cross-encoder re-rank
└────────┬────────┘
         │
    enough context?
    ├── YES ──▶ Synthesizer ──▶ Streamed Answer + Citations
    │
    └── NO  ──▶ Query Rewriter ──▶ Retriever  (max 3 attempts)
                                         │
                                         └── still nothing ──▶ Graceful fallback
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM & Embeddings | Ollama (`mistral:latest`, `nomic-embed-text`) via LiteLLM |
| Vector Database | ChromaDB (persistent, on-disk) |
| Re-ranking | sentence-transformers cross-encoder |
| Backend | FastAPI + uvicorn, Python 3.11 |
| Frontend | React, Vite, Vanilla CSS (Glassmorphism UI) |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions (lint + test + Docker build) |
| Evaluation | RAGAS (context precision, recall, faithfulness, answer relevancy) |

## 🚀 Getting Started

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/) & Docker Compose  
  **or** Python 3.11+ and [Ollama](https://ollama.com/) installed locally.

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/yourusername/academic-rag-pipeline.git
cd academic-rag-pipeline
docker-compose up --build
```

> **Note:** Ollama will pull `mistral` and `nomic-embed-text` on first run — allow a few minutes.

### Option 2: Local (without Docker)

```bash
# 1. Pull the required Ollama models
ollama run mistral
ollama pull nomic-embed-text

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux
pip install -r requirements.txt

# 3. Copy and configure the environment file
copy .env.example .env        # Windows
# cp .env.example .env        # Mac / Linux

# 4. Index academic papers into ChromaDB (run once)
python -m src.main --build-store

# 5. Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` in your browser for the API documentation.

### Starting the React Frontend

To use the modern chat UI, start the frontend development server:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. Make sure the backend (Docker or local `uvicorn`) is running.

## 📁 Project Structure

```text
.
├── .github/workflows/ci.yml  # GitHub Actions: lint → test → Docker build
├── data/                     # corpus.json (PubMed papers) and raw PDFs
├── frontend/                 # React + Vite frontend application
├── scripts/
│   └── evaluate_ragas.py     # Offline RAGAS evaluation script
├── src/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine definition
│   │   ├── model.py          # LLMFactory (LiteLLM wrapper, singleton)
│   │   ├── state.py          # RAGState TypedDict
│   │   └── nodes/
│   │       ├── query_analyzer.py      # Scope classification + sub-query decomposition
│   │       ├── retriever_evaluator.py # Hybrid retrieval + cross-encoder re-ranking
│   │       ├── synthesizer.py         # Citation-grounded answer generation
│   │       └── query_rewriter.py      # Query reformulation for retry
│   ├── config/settings.py    # Pydantic settings (reads from .env)
│   ├── embeddings/embedder.py # OllamaEmbedder (singleton, batch support)
│   ├── ingestion/
│   │   ├── fetch_papers.py   # PubMed E-utilities scraper
│   │   └── loader.py         # corpus.json / PDF / TXT loader
│   ├── rag/pipeline.py       # build_vector_store() orchestrator
│   ├── splitting/splitter.py # Token-based chunking (tiktoken / whitespace)
│   ├── vectordb/chroma_store.py # ChromaDB wrapper (hybrid search, retry)
│   ├── api.py                # FastAPI endpoints + SSE streaming + semantic cache
│   └── main.py               # CLI entrypoint (--build-store or interactive loop)
├── tests/                    # Pytest suite — 27 tests, 100% pass rate
├── .env.example              # Template for environment variables
├── Dockerfile                # Multi-stage container build
├── docker-compose.yml        # App + Ollama service orchestration
└── requirements.txt          # All Python dependencies with comments
```

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

27 tests covering the API endpoints, vector store, query analyzer, graph pipeline, and chunking logic.

## 📊 Evaluation

To evaluate pipeline quality with RAGAS metrics (requires Ollama running):

```bash
python scripts/evaluate_ragas.py
```

Metrics: **context_precision**, **context_recall**, **faithfulness**, **answer_relevancy**.

## 🛣️ Potential Future Improvements

* **Persistent chat memory** — store conversation history in a database for long sessions.
* **Multi-document corpora** — extend ingestion beyond ALS to other biomedical domains.
* **Authentication** — add API key or OAuth protection to the `/query` endpoint.
* **Streaming citations** — stream citation data progressively alongside tokens.

---
*Built as a portfolio project demonstrating advanced LLM orchestration, local AI deployment, agentic architectures, and robust Python backend engineering.*
