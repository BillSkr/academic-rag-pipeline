"""Tests for the Query Analyzer node.

The query analyzer is the first node in the RAG pipeline graph.
It is responsible for:
  1. Rejecting empty or whitespace-only queries immediately.
  2. Classifying the query as academic (YES) or off-topic (NO) via the LLM.
  3. If academic, decomposing the query into sub-queries via a second LLM call.

The LLMFactory is mocked so no real model API calls are made.
"""

import pytest
from unittest.mock import patch

from src.agent.nodes.query_analyzer import analyze_query


# ──────────────────────────── Helper ─────────────────────────────────────────

def _make_state(query: str) -> dict:
    """Return a minimal RAGState dict for a given user query."""
    return {"user_query": query, "rejected": False}


# ──────────────────────────── Tests ──────────────────────────────────────────

@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_rejected(query):
    """Empty or whitespace-only queries must be rejected before any LLM call."""
    state = _make_state(query)
    analyze_query(state)

    # Should be marked rejected with a user-friendly message
    assert state["rejected"] is True
    assert state.get("response") == "Please enter a valid question."


@pytest.mark.parametrize(
    "query",
    [
        "What causes ALS?",
        "Latest clinical trials for motor neuron disease",
        "Research on neurodegenerative diseases",
    ],
)
@patch("src.agent.nodes.query_analyzer.LLMFactory")
def test_academic_query_accepted(mock_llm_cls, query):
    """Academic queries should be accepted, sub-queries parsed, and state updated."""
    # LLM calls: (1) YES/NO classification, (2) JSON sub-query decomposition
    mock_instance = mock_llm_cls.return_value
    mock_instance.generate.side_effect = [
        "YES",
        '{"sub_queries": ["What is ALS?"]}',
    ]

    state = _make_state(query)
    analyze_query(state)

    assert state["rejected"] is False
    assert state["current_query"] == query
    assert len(state["sub_queries"]) == 1
    # Exactly 2 LLM calls should have been made
    assert mock_instance.generate.call_count == 2


@pytest.mark.parametrize(
    "query",
    [
        "Best pizza in Rome",
        "How to train a dog",
        "Top 10 football players of 2024",
    ],
)
@patch("src.agent.nodes.query_analyzer.LLMFactory")
def test_non_academic_query_rejected(mock_llm_cls, query):
    """Off-topic queries should be rejected after a single LLM classification call."""
    mock_instance = mock_llm_cls.return_value
    mock_instance.generate.return_value = "NO"  # classification returns NO

    state = _make_state(query)
    analyze_query(state=state)

    assert state["rejected"] is True
    assert "not appear to be related to academic research" in state["response"]
    # Only 1 LLM call (classification) — no decomposition for rejected queries
    assert mock_instance.generate.call_count == 1


@patch("src.agent.nodes.query_analyzer.LLMFactory")
def test_short_valid_query_accepted(mock_llm_cls):
    """A short but topically valid query should also pass the classifier."""
    mock_instance = mock_llm_cls.return_value
    mock_instance.generate.side_effect = [
        "YES",
        '{"sub_queries": ["What is ALS treatment?"]}',
    ]

    state = _make_state("ALS treatment")
    analyze_query(state=state)

    assert state["rejected"] is False
    assert state["current_query"] == "ALS treatment"
    assert len(state["sub_queries"]) == 1