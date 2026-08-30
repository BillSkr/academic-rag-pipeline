"""FastAPI web service for the Academic RAG Assistant.

Exposes two endpoints:
    POST /query      – submit a question and receive an answer with citations.
    POST /build      – trigger a vector-store rebuild (async).
"""

import asyncio
import json
import math
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.graph import create_rag_graph
from src.agent.state import RAGState
from src.embeddings.embedder import OllamaEmbedder
from src.rag.pipeline import build_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple In-Memory Semantic Cache (capped to avoid unbounded memory growth)
_semantic_cache: list = []
_MAX_CACHE_SIZE = 100


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


app = FastAPI(title="Academic RAG Assistant")

# CORS middleware so the frontend (index.html) can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = create_rag_graph()


class QueryRequest(BaseModel):
    question: str
    history: list = Field(default_factory=list)


@app.post("/query")
async def query(request: QueryRequest):
    """Run the RAG pipeline on a user question, streaming results via SSE."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    user_query = request.question.strip()

    async def event_generator():
        # Yield an initial status immediately so the frontend knows we are working
        yield f"data: {json.dumps({'status': 'Analyzing query...'})}\n\n"

        query_embedding = None
        try:
            embedder = OllamaEmbedder()
            # Run the blocking embed call in a thread so it doesn't freeze the event loop
            query_embedding = await asyncio.to_thread(embedder.embed, user_query)
            
            # Check cache for similar query
            for cached in _semantic_cache:
                if cosine_similarity(query_embedding, cached["embedding"]) > 0.95:
                    logger.info("Cache hit for similar query")
                    yield f"data: {json.dumps({'status': 'completed', 'response': cached['response'], 'citations': cached['citations']})}\n\n"
                    return
        except Exception as e:
            logger.error(f"Embedding failed for cache check: {e}")

        state: RAGState = {
            "chat_history": request.history,
            "user_query": user_query,
            "sub_queries": [],
            "current_query": "",
            "retrieved_chunks": [],
            "attempts": 0,
            "response": "",
            "citations": [],
            "rejected": False,
            "enough": False,
        }

        try:
            # Run graph synchronously in a thread with a timeout
            logger.info(f"Starting graph execution for query: {user_query}")
            final_state = await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, state),
                timeout=180  # 3 minute timeout
            )
            logger.info(f"Graph execution completed")
            
            response_text = final_state.get("response", "")
            citations = final_state.get("citations", [])
            
            logger.info(f"Response length: {len(response_text)}, Citations: {len(citations)}")
            
            # Save to cache if we got a valid response (evict oldest if at cap)
            if not final_state.get("rejected") and query_embedding and response_text:
                if len(_semantic_cache) >= _MAX_CACHE_SIZE:
                    _semantic_cache.pop(0)
                _semantic_cache.append({
                    "embedding": query_embedding,
                    "response": response_text,
                    "citations": citations
                })
            
            # Yield the completed status with response and citations
            yield f"data: {json.dumps({'status': 'completed', 'response': response_text, 'citations': citations})}\n\n"
        except asyncio.TimeoutError:
            logger.error("Graph execution timed out")
            yield f"data: {json.dumps({'status': 'completed', 'response': 'Request timed out. Please try again.', 'citations': []})}\n\n"
        except Exception as e:
            logger.error(f"Graph execution error: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'completed', 'response': f'Error: {str(e)}', 'citations': []})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/build")
async def build():
    """Rebuild the vector store from source documents."""
    try:
        build_vector_store()
        return {"status": "Vector store rebuilt successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Build failed: {exc}") from exc


@app.get("/health")
async def health():
    """Health check for Render (and load balancers)."""
    return {"status": "ok"}
