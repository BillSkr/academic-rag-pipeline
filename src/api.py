"""FastAPI web service for the Academic RAG Assistant.

Exposes two endpoints:
    POST /query      – submit a question and receive an answer with citations.
    POST /build      – trigger a vector-store rebuild (async).
"""

import asyncio
import json
import math

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.graph import create_rag_graph
from src.agent.state import RAGState
from src.embeddings.embedder import OllamaEmbedder
from src.rag.pipeline import build_vector_store

# Simple In-Memory Semantic Cache
_semantic_cache = []

def cosine_similarity(v1, v2):
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


def _extract_final_state(graph_output: dict) -> dict:
    """Return the RAG state from either flat or node-wrapped LangGraph output."""
    if not isinstance(graph_output, dict):
        return {}
    if "response" in graph_output:
        return graph_output
    for value in graph_output.values():
        if isinstance(value, dict) and "response" in value:
            return value
    return {}


class QueryRequest(BaseModel):
    question: str
    history: list = []


@app.post("/query")
async def query(request: QueryRequest):
    """Run the RAG pipeline on a user question."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    user_query = request.question.strip()

    # --- IMPROVEMENT 2: UI/UX Streaming and Agent Visibility ---
    async def event_generator():
        # Yield an initial status immediately so the frontend knows we are working
        yield f"data: {json.dumps({'status': 'Analyzing query (loading models)...'})}\n\n"

        # --- IMPROVEMENT 5: Semantic Caching (Performance Optimization) ---
        query_embedding = None
        try:
            embedder = OllamaEmbedder()
            # Run the blocking embed call in a thread so it doesn't freeze the event loop
            query_embedding = await asyncio.to_thread(embedder.embed, user_query)
            
            # Check cache for similar query
            for cached in _semantic_cache:
                if cosine_similarity(query_embedding, cached["embedding"]) > 0.95:
                    yield f"data: {json.dumps({'status': 'completed', 'response': cached['response'], 'citations': cached['citations']})}\n\n"
                    return
        except Exception as e:
            print(f"Embedding failed for cache check: {e}")

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
            async for event in graph.astream_events(state, version="v1"):
                event_type = event.get("event")
                
                # Stream status updates for node transitions
                if event_type == "on_chain_start" and event.get("name") != "LangGraph":
                    node_name = event.get("name")
                    yield f"data: {json.dumps({'status': f'Running {node_name}...'})}\n\n"
                    
                # Stream LLM tokens from the synthesizer
                elif event_type == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content') and chunk.content:
                        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                        
                # Complete and cache
                elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = _extract_final_state(event["data"]["output"])
                    # Save to cache
                    if not final_state.get("rejected") and query_embedding:
                        _semantic_cache.append({
                            "embedding": query_embedding,
                            "response": final_state.get("response", ""),
                            "citations": final_state.get("citations", [])
                        })
                    yield f"data: {json.dumps({'status': 'completed', 'response': final_state.get('response', ''), 'citations': final_state.get('citations', [])})}\n\n"
        except Exception as e:
            print(f"Graph execution error: {e}")
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
