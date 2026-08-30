"""
Node 2 – Retriever & Evaluator.

Responsibilities:
  * Embed the current search query using OllamaEmbedder.
  * Run the query against the ChromaDB vector store for all sub-queries.
  * Deduplicate results and filter by similarity threshold.
  * Re-rank the candidates with a cross-encoder (ms-marco-MiniLM-L-6-v2).
  * Set state["enough"] = True if any chunks pass the threshold.
"""

import logging

from sentence_transformers import CrossEncoder

from src.agent.state import RAGState
from src.config import settings
from src.embeddings.embedder import OllamaEmbedder
from src.vectordb.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def retrieve_and_evaluate(state: "RAGState") -> "RAGState":
    """Retrieve candidate chunks and decide whether they are sufficient."""

    logger.info("Retrieving chunks for query: %s", state["current_query"])

    embedder = OllamaEmbedder()
    store = ChromaVectorStore()

    # Run retrieval for the primary query and all decomposed sub-queries
    queries_to_run = [state["current_query"]]
    if state.get("sub_queries"):
        queries_to_run.extend(state["sub_queries"])

    # Use a dict keyed by chunk ID to deduplicate across all queries
    all_chunks_dict: dict = {}

    for q in queries_to_run:
        try:
            q_vec = embedder.embed(q)
            q_chunks = store.similarity_search(q_vec, k=settings.TOP_K)

            # Only keep chunks within the configured similarity threshold
            for c in q_chunks:
                if c["distance"] <= settings.SIMILARITY_THRESHOLD:
                    all_chunks_dict[c["id"]] = c
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", q, exc)

    chunks = list(all_chunks_dict.values())
    logger.info(f"Retrieved {len(chunks)} chunks from vector store")

    # If nothing was retrieved, signal that retrieval failed
    if not chunks:
        logger.warning("No chunks retrieved - setting enough=False to trigger retry")
        state["retrieved_chunks"] = []
        state["enough"] = False
        return state

    # ── Cross-encoder re-ranking ──────────────────────────────────────────────
    # The cross-encoder scores (query, chunk) pairs more accurately than
    # cosine similarity and re-orders the candidates accordingly.
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates = [(state["current_query"], c["document"]) for c in chunks]
    rerank_scores = cross_encoder.predict(rerank_candidates)

    # Sort descending by cross-encoder score
    scored_chunks = sorted(zip(rerank_scores, chunks), key=lambda x: x[0], reverse=True)
    chunks = [chunk for _, chunk in scored_chunks]

    state["retrieved_chunks"] = chunks
    state["enough"] = True   # we have at least one valid chunk

    return state