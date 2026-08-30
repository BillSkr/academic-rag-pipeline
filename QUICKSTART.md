# Quick Start Guide - RAG Pipeline for Portfolio

## 5-Minute Setup

### Step 1: Start Everything
```bash
cd "RAG pipeline"
docker compose up -d --pull always
```

Wait for services to start (~2 minutes). Check status:
```bash
docker compose ps
```

All services should show "Up":
- `ollama-service` (LLM)
- `rag-app` (API server)
- `rag-frontend` (Web UI)

### Step 2: Build the Vector Store
Before querying, you must index the academic papers into the database:
```bash
docker exec rag-app python -m src.main --build-store
```
Wait for the "Vector store built successfully" message.

### Step 3: Test it Works
```bash
# Quick health check (should return {"status":"ok"})
curl http://localhost:8001/health
```

### Step 4: Try Queries

**Option A: Web Interface (Easiest)**
Open browser: http://localhost:4173
Type any academic question and wait for response.

**Option B: Command Line (Python)**
```python
import requests
import json

r = requests.post(
    'http://localhost:8001/query',
    json={'question': 'What is SOD1 protein?'},
    timeout=180,
    stream=True
)

for line in r.iter_lines():
    if line and b'completed' in line:
        data = json.loads(line.decode().replace('data: ', ''))
        print("Answer:", data['response'][:200])
        print("Sources:", len(data['citations']))
        break
```

**Option C: Shell Script**
```bash
bash scripts/test_query.sh "What causes Alzheimer's?"
```

---

## Try These Queries

### ✅ Works (Academic)
- "What is CRISPR and how does it work?"
- "Explain SOD1 protein"
- "What is amyotrophic lateral sclerosis?"

### ❌ Gets Rejected (Non-academic)
- "What's your favorite color?"
- "How do I make pizza?"

---

## For Your Portfolio

### Screenshots to Include
1. Frontend showing a query
2. API response with citations
3. Performance metrics
4. Docker Compose setup

### README Talking Points
- "Built a complete RAG system with local LLMs"
- "Implemented agentic query reformulation"
- "Integrated ChromaDB for semantic search"
- "Designed real-time SSE streaming"
- "Deployed with Docker Compose"

### Live Demo
Can run on a cheap server ($5-10/month):
- AWS EC2 t3.medium
- DigitalOcean Droplet
- Linode 4GB

---

## Troubleshooting

### "Connection refused"
Container not ready yet. Check logs:
```bash
docker logs rag-app
```

### "Only getting fallback responses"
Vector store might not have relevant data. Verify:
```bash
docker exec rag-app python -c "
from src.vectordb.chroma_store import ChromaVectorStore
store = ChromaVectorStore()
print(f'Documents in store: {store.collection.count()}')
"
```

### "Very slow first query"
Normal! Ollama needs to load models (~60-90 seconds).
Subsequent queries are cached and faster.

---

## Next Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial RAG pipeline"
   git remote add origin <your-repo>
   git push origin main
   ```

2. **Add to Portfolio**
   - Create project card with description
   - Link to GitHub repo
   - Add screenshots
   - Explain tech stack

3. **Deploy Live (Optional)**
   - Rent a server
   - Run `docker compose up -d`
   - Share public URL with recruiters

4. **Document Experience**
   - "Designed & deployed full-stack RAG system"
   - "Integrated LangGraph for agentic workflows"
   - "Optimized retrieval with ChromaDB & re-ranking"

---

## Commands Reference

```bash
# View logs
docker logs rag-app -f

# Restart services
docker compose restart

# Stop everything
docker compose down

# Clean up (removes volumes)
docker compose down -v

# Run tests
python tests/test_api.py

# Rebuild image
docker compose build

# SSH into container
docker exec -it rag-app bash
```

That's it! You're ready to showcase this project. 🚀
