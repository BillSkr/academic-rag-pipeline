# Academic RAG Pipeline - Portfolio Project

A production-ready Retrieval-Augmented Generation (RAG) system for academic research papers, built with Groq (Llama 3), LangGraph, and ChromaDB.

## 🎯 Project Overview

This project demonstrates a complete AI pipeline that:
- **Retrieves** relevant research papers from a vector database
- **Augments** queries with domain-specific context
- **Generates** accurate, cited answers using local LLMs
- **Streams** real-time responses to the frontend
- **Caches** results for performance optimization

### Key Features
- ✅ Lightning-fast cloud LLM inference via Groq API
- ✅ Agentic query reformulation for better retrieval
- ✅ Cross-encoder re-ranking for result quality
- ✅ Server-Sent Events (SSE) streaming
- ✅ Semantic caching for repeated queries
- ✅ Production-grade error handling
- ✅ Full Docker containerization
- ✅ React frontend with real-time updates

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Groq API Key (Set in `.env`)
- ~1GB RAM minimum

### 1. Clone & Setup
```bash
git clone <your-repo>
cd RAG\ pipeline
```

### 2. Start the Stack
```bash
docker compose up -d --pull always
```

The first startup may download the lightweight local embedding model (~100MB):
- **all-MiniLM-L6-v2** (384-dim embeddings)
- **cross-encoder** (for re-ranking)

### 3. Build the Vector Store
Before querying, index the academic papers into the database:
```bash
docker exec rag-app python -m src.main --build-store
```

### 4. Access the Application
- **Frontend**: http://localhost:4173
- **API**: http://localhost:8001


---

## 🧪 Testing & Interaction

### Test 1: Health Check
```bash
curl http://localhost:8001/health
```
Expected: `{"status":"ok"}`

### Test 2: Query the RAG Pipeline (Python)
```python
import requests

# Example academic query
response = requests.post(
    'http://localhost:8001/query',
    json={'question': 'What is SOD1 protein and its role in ALS?'},
    timeout=180,
    stream=True
)

for line in response.iter_lines():
    if line and b'completed' in line:
        data = json.loads(line.decode().replace('data: ', ''))
        print(f"Answer: {data['response']}")
        print(f"Citations: {len(data['citations'])} sources")
        break
```

### Test 3: Benchmark Performance
Run the included test suite:
```bash
python tests/benchmark.py
```

This measures:
- Query latency
- Cache hit rate
- Token throughput
- Memory usage

### Test 4: Interactive Web Interface
1. Open http://localhost:4173
2. Type: `"What causes Alzheimer's disease?"`
3. Watch real-time streaming responses
4. View citations with full metadata

---

## 📊 API Documentation

### POST /query
Submit a question to the RAG pipeline.

**Request:**
```json
{
  "question": "What is SOD1 protein?",
  "history": []
}
```

**Response (SSE Stream):**
```
data: {"status":"Analyzing query..."}
data: {"status":"Running analyzer..."}
data: {"status":"Running retriever..."}
data: {"status":"completed","response":"...","citations":[...]}
```

**Response Fields:**
- `status` (string): Pipeline stage or "completed"
- `response` (string): Final answer with citations
- `citations` (array): Retrieved document chunks with metadata

### POST /build
Rebuild the vector store from source documents.

**Response:**
```json
{"status": "Vector store rebuilt successfully."}
```

### GET /health
Health check endpoint.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                      │
│            (Vite + Real-time SSE)                   │
└────────────────┬────────────────────────────────────┘
                 │ HTTP/SSE
┌────────────────▼────────────────────────────────────┐
│              FastAPI Backend                         │
│         (Port 8000, Exposed on 8001)                │
├──────────────────────────────────────────────────────┤
│  1. Semantic Caching Layer                          │
│  2. Query Embedding (LocalEmbedder)                 │
│  3. LangGraph Agentic Pipeline                      │
│     ├─ Query Analyzer (classify & decompose)       │
│     ├─ Retriever & Evaluator (vector search)       │
│     ├─ Query Rewriter (reformulate if needed)      │
│     └─ Synthesizer (generate citations)            │
│  4. Error Handling & Logging                        │
└────────────────┬────────────────────────────────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
┌─────▼───┐  ┌──▼────┐  ┌──▼──────────────┐
│ Groq    │  │Chroma │  │cross-encoder/  │
│ (Llama3)│  │ DB    │  │ms-marco        │
│ (LLM)   │  │(Vec)  │  │(Re-ranker)     │
└─────────┘  └───────┘  └─────────────────┘
```

---

## 📈 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **First Query** | 5-10s | Model load (if needed) |
| **Cached Query** | 1-2s | Semantic cache hit |
| **Retrieval** | 1-2s | Vector search + re-ranking |
| **LLM Inference** | 2-5s | Groq API generation |
| **Memory Usage** | ~1GB | Container limits set |

---

## 🛠️ Development

### Project Structure
```
RAG pipeline/
├── src/
│   ├── api.py                 # FastAPI application
│   ├── agent/                 # LangGraph nodes
│   │   ├── graph.py           # Pipeline orchestration
│   │   ├── state.py           # Shared state definition
│   │   └── nodes/             # Individual processing nodes
│   ├── embeddings/            # LocalEmbedder
│   ├── vectordb/              # ChromaDB wrapper
│   ├── rag/                   # RAG pipeline logic
│   └── config/                # Settings & configuration
├── frontend/                  # React + Vite
├── data/                      # Research papers
├── tests/                     # Test suite & benchmarks
├── docker-compose.yml         # Multi-service orchestration
├── Dockerfile                 # Python app
└── requirements.txt           # Python dependencies
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end benchmark
python tests/benchmark.py --queries 10 --save-results
```

### Local Development (without Docker)
```bash
pip install -r requirements.txt

python -m uvicorn src.api:app --reload  # In another
```

---

## 🔑 Key Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Groq API + Llama 3 | Lightning fast, cloud-hosted, easy to deploy |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim, lightweight, CPU-friendly |
| **Vector DB** | ChromaDB | Simple, persistent, Python-native |
| **Agentic Loop** | LangGraph | Clean state management, composable |
| **Web Framework** | FastAPI | Async, SSE streaming, auto-docs |
| **Frontend** | React + Vite | Real-time updates, modern tooling |
| **Containerization** | Docker Compose | Reproducible, multi-service |

---

## 📝 Sample Test Queries

### Academic Questions (Works Well)
- "What is CRISPR and how does it work?"
- "Explain the role of dopamine in Parkinson's disease"
- "What are the latest advances in quantum computing?"

### Non-Academic (Rejected)
- "What's the best pizza topping?"
- "How do I cook pasta?"
- "Tell me a joke"

---

## 🚨 Troubleshooting

### API Returns "No Results Found"
- **Cause**: Query doesn't match any documents OR similarity threshold too strict
- **Fix**: Increase `SIMILARITY_THRESHOLD` in `src/config/settings.py` (e.g., to 0.8)
- **Test**: `curl http://localhost:8001/query -X POST -H "Content-Type: application/json" -d '{"question":"SOD1 protein"}'`

### Frontend Shows Loading Spinner Forever
- **Cause**: Request timeout
- **Fix**: Check network connectivity to Groq API
- **Check**: `curl -w "@curl-format.txt" http://localhost:8001/health`



---

## 🎓 Portfolio Highlights

This project demonstrates:
✅ **Full-stack development** (Python backend, React frontend)
✅ **LLM orchestration** (LangGraph, agent loops)
✅ **Vector databases** (embeddings, retrieval)
✅ **Real-time streaming** (SSE, async Python)
✅ **Production practices** (error handling, caching, logging)
✅ **DevOps** (Docker, multi-service orchestration)
✅ **API design** (FastAPI, REST principles)
✅ **Performance optimization** (semantic caching, re-ranking)

---

## 📜 License

MIT - Feel free to use this for your portfolio

## 👨‍💻 Author

Built as a demonstration of RAG systems and AI engineering practices.

---

**Ready to showcase?** 
1. Host on GitHub
2. Add live demo link (if running on a server)
3. Include screenshots in README
4. Reference this in your CV/Portfolio under "AI/ML Projects"
