"""Integration tests for the full RAG LangGraph pipeline.

These tests verify that the graph can execute all nodes end-to-end
using mocked external dependencies (LLM, embedder, vector store, cross-encoder).

The graph flow is:
    analyze_query → retrieve_and_evaluate → synthesize_answer
"""

from unittest.mock import patch, MagicMock
from src.agent.graph import create_rag_graph


# ──────────────────────────── Helpers ────────────────────────────────────────

def _make_initial_state(query: str = "What is the role of SOD1 in ALS?") -> dict:
    """Return a clean initial RAGState dict for test runs."""
    return {
        "user_query": query,
        "chat_history": [],
        "sub_queries": [],
        "current_query": "",
        "retrieved_chunks": [],
        "attempts": 0,
        "response": "",
        "citations": [],
        "rejected": False,
        "enough": False,
    }


# ──────────────────────────── Tests ──────────────────────────────────────────

@patch("src.agent.nodes.retriever_evaluator.OllamaEmbedder")
@patch("src.agent.nodes.retriever_evaluator.ChromaVectorStore")
@patch("src.agent.nodes.retriever_evaluator.CrossEncoder")
@patch("src.agent.nodes.query_analyzer.LLMFactory")    # Patch the class, not the method
@patch("src.agent.nodes.synthesizer.LLMFactory")       # Patch the class, not the method
def test_full_graph_execution(
    mock_synth_llm_cls,
    mock_analyzer_llm_cls,
    mock_cross_encoder_cls,
    mock_vector_store_cls,
    mock_embedder_cls,
):
    """Test that the graph executes all nodes end-to-end without crashing.

    Mocking strategy:
    - LLMFactory is patched at the class level so that LLMFactory() inside each
      node returns a fully controlled MagicMock instance.
    - OllamaEmbedder and ChromaVectorStore are also class-patched so their
      constructors return mock instances.
    """

    # ── Query Analyzer mock ──────────────────────────────────────────────────
    # generate() is called twice: once for YES/NO classification, once for JSON decomposition
    analyzer_instance = mock_analyzer_llm_cls.return_value
    analyzer_instance.generate.side_effect = [
        "YES",                                      # 1st call: is it academic?
        '{"sub_queries": ["Sub-query 1"]}',         # 2nd call: decompose into sub-queries
    ]

    # ── Retriever / Evaluator mocks ──────────────────────────────────────────
    # Embedder returns a fixed 384-dim vector
    embedder_instance = mock_embedder_cls.return_value
    embedder_instance.embed.return_value = [0.1] * 384

    # Vector store returns one plausible chunk well below the similarity threshold
    store_instance = mock_vector_store_cls.return_value
    store_instance.similarity_search.return_value = [
        {
            "id": "1",
            "document": "Mock document about SOD1 mutations.",
            "distance": 0.1,   # low distance = high similarity
            "metadata": {"title": "Test Paper", "year": "2023"},
        }
    ]

    # Cross-encoder assigns a high re-ranking score to the chunk
    cross_encoder_instance = mock_cross_encoder_cls.return_value
    cross_encoder_instance.predict.return_value = [0.9]

    # ── Synthesizer mock ─────────────────────────────────────────────────────
    synth_instance = mock_synth_llm_cls.return_value
    synth_instance.generate.return_value = "This is a mock final answer."

    # ── Run the graph ────────────────────────────────────────────────────────
    graph = create_rag_graph()
    final_state = graph.invoke(_make_initial_state())

    # ── Assertions ───────────────────────────────────────────────────────────
    assert final_state["rejected"] is False
    assert final_state["response"] == "This is a mock final answer."
    assert len(final_state["citations"]) == 1
    assert final_state["sub_queries"] == ["Sub-query 1"]
