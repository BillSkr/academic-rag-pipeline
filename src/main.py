"""
Entry point for the Academic RAG Assistant.

Two possible execution modes:

1. **Build vector store** – run once (or when new documents are added):
       python -m src.main --build-store

2. **Start interactive query loop** – feed questions to the LangGraph pipeline:
       python -m src.main
"""

import argparse
import os

from dotenv import load_dotenv

from src.agent.graph import create_rag_graph
from src.agent.state import RAGState  # RAGState lives in state.py, not graph.py
from src.rag.pipeline import build_vector_store

# Load .env file before accessing any environment variables
load_dotenv()

# Optional: enable LangChain / LangSmith tracing when the API key is present
_langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
if _langchain_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = _langchain_key
    os.environ["LANGCHAIN_PROJECT"] = "RAG-Trace"


# ──────────────────────────── Helpers ────────────────────────────────────────

def _fresh_state(query: str) -> RAGState:
    """Return a clean initial state dict for a single query."""
    return {
        "user_query": query,
        "chat_history": [],       # conversation history (empty for CLI mode)
        "sub_queries": [],
        "current_query": "",
        "retrieved_chunks": [],
        "attempts": 0,
        "response": "",
        "citations": [],
        "rejected": False,
        "enough": False,
    }


def _print_citations(citations: list) -> None:
    """Pretty-print citation metadata to stdout."""
    if not citations:
        print("No citations available.")
        return
    print("\nCitations:")
    for citation in citations:
        meta = citation.get("metadata", {})
        title = meta.get("title", "Unknown")
        year = meta.get("year", "N/A")
        print(f"  - {title} ({year})")


# ──────────────────────────── Main ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic RAG Assistant")
    parser.add_argument(
        "--build-store",
        action="store_true",
        help="Populate the Chroma vector store from source documents, then exit.",
    )
    args = parser.parse_args()

    if args.build_store:
        # One-shot indexing mode
        try:
            build_vector_store()
            print("Vector store built successfully.")
        except Exception as exc:
            print(f"Failed to build vector store: {exc}")
    else:
        # Interactive query loop
        graph = create_rag_graph()
        print("RAG session started. Type 'exit' or 'quit' to leave.\n")

        while True:
            query = input("Query > ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            try:
                final_state = graph.invoke(_fresh_state(query))
            except Exception as exc:
                print(f"Error processing query: {exc}\n")
                continue

            if final_state.get("rejected"):
                print(f"  {final_state.get('response', 'Query rejected.')}\n")
                continue

            print(f"\n{final_state.get('response', '')}\n")
            _print_citations(final_state.get("citations", []))
            print()