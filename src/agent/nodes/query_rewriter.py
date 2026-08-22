"""
Node 4 – Query Rewriter.

Triggered when the retriever fails to find enough relevant chunks.
Uses the LLM to reformulate the search query using synonyms,
broader terms, or more specific disease/treatment terminology
to improve the next retrieval attempt.
"""

import logging

from src.agent.state import RAGState
from src.agent.model import LLMFactory

logger = logging.getLogger(__name__)


def rewrite_query(state: "RAGState") -> "RAGState":
    """Rewrite the current query to improve vector retrieval on the next attempt."""
    model = LLMFactory()

    original_query = state.get("user_query", "")
    current_query = state.get("current_query", original_query)

    prompt = (
        f"Original User Question: {original_query}\n"
        f"Previous Search Query: {current_query}\n\n"
        "The previous search query did not return any relevant academic papers. "
        "Please rewrite the search query to be broader, use synonyms, or focus on the core concepts "
        "(like specific disease names or general treatment classes) "
        "to improve the chances of finding relevant medical/academic literature.\n"
        "Return ONLY the new query string, without any quotes or extra explanation."
    )

    new_query = model.generate(
        system_prompt=(
            "You are a helpful assistant optimizing search queries for a vector database of academic papers. "
            "Output exactly the optimized query string and nothing else."
        ),
        user_prompt=prompt,
    )

    # Strip surrounding whitespace and quotes the LLM may accidentally add
    new_query = new_query.strip().strip('"').strip("'")

    state["current_query"] = new_query
    logger.info("[Rewriter] Reformulated query: %s", new_query)
    return state
