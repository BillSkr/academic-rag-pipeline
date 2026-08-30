# Academic RAG Pipeline - Portfolio Project

A production-ready Retrieval-Augmented Generation (RAG) system for academic research papers, built with local LLMs (Ollama), LangGraph, and ChromaDB.

## 🎯 Project Overview

This project demonstrates a complete AI pipeline that:
- **Retrieves** relevant research papers from a vector database
- **Augments** queries with domain-specific context
- **Generates** accurate, cited answers using local LLMs
- **Streams** real-time responses to the frontend
- **Caches** results for performance optimization

### Key Features
- ✅ Local LLM inference (no API keys required)
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
- 8GB RAM minimum (16GB recommended)
- ~50GB disk space (for Ollama models)

### 1. Clone & Setup
```bash
git clone <your-repo>
cd RAG\ pipeline
```

### 2. Start the Stack
```bash
docker compose up -d --pull always
```

The first startup downloads models (~30 minutes):
- **Ollama Mistral** (7B LLM for inference)
- **nomic-embed-text** (384-dim embeddings)
- **cross-encoder** (for re-ranking)

### 3. Access the Application
- **Frontend**: http://localhost:4173
- **API**: http://localhost:8001
- **Ollama**: http://localhost:11435

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
data: {"status":"Analyzing query (loading models)..."}
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
│  2. Query Embedding (OllamaEmbedder)                │
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
│ Ollama  │  │Chroma │  │cross-encoder/  │
│ Mistral │  │ DB    │  │ms-marco        │
│ (LLM)   │  │(Vec)  │  │(Re-ranker)     │
└─────────┘  └───────┘  └─────────────────┘
```

---

## 📈 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **First Query** | 90-120s | LLM cold start |
| **Cached Query** | 2-5s | Semantic cache hit |
| **Retrieval** | 1-2s | Vector search + re-ranking |
| **LLM Inference** | 30-60s | Mistral 7B token generation |
| **Memory Usage** | ~8GB | Container limits set |

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
│   ├── embeddings/            # OllamaEmbedder
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
ollama serve  # In one terminal
python -m uvicorn src.api:app --reload  # In another
```

---

## 🔑 Key Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Ollama + Mistral 7B | Open-source, local, no API costs |
| **Embeddings** | nomic-embed-text | 384-dim, efficient, purpose-built |
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
- **Cause**: Query doesn't match any documents OR similarity threshold too high
- **Fix**: Lower `SIMILARITY_THRESHOLD` in `src/config/settings.py` from 0.5 to 0.3
- **Test**: `curl http://localhost:8001/query -X POST -H "Content-Type: application/json" -d '{"question":"SOD1 protein"}'`

### Frontend Shows Loading Spinner Forever
- **Cause**: Request timeout (queries take 90-120s)
- **Fix**: Increase browser timeout or wait longer
- **Check**: `curl -w "@curl-format.txt" http://localhost:8001/health`

### Out of Memory
- **Cause**: Models loaded exceed 8GB
- **Fix**: Increase Docker memory in `docker-compose.yml`
```yaml
rag-app:
  mem_limit: 16g  # Increase from 8g
```

### Models Not Downloading
- **Cause**: First startup takes 30 minutes
- **Status**: Check with `docker logs ollama-service`
- **Wait**: Let it complete before querying

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
