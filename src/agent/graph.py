"""
LangGraph graph definition.

Registers the four core nodes and wires up the conditional edges
that allow iterative query reformulation when retrieval fails.

Pipeline flow:
    analyze_query → (rejected? END : retrieve_and_evaluate)
                  → (enough? synthesize_answer : loop_or_end)
                  → (attempts < MAX? rewrite_query → retrieve_and_evaluate : END)
"""

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    query_analyzer,
    query_rewriter,
    retriever_evaluator,
    synthesizer,
)
from src.agent.state import RAGState
from src.config import settings


def create_rag_graph():
    """Build and compile the LangGraph state machine.

    Returns a compiled graph ready to be invoked with an initial RAGState.
    """
    workflow = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("analyzer", query_analyzer.analyze_query)
    workflow.add_node("retriever", retriever_evaluator.retrieve_and_evaluate)
    workflow.add_node("synthesizer", synthesizer.synthesize_answer)
    workflow.add_node("rewriter", query_rewriter.rewrite_query)

    # ── Loop-or-end decision node ─────────────────────────────────────────────
    def loop_or_end(state: RAGState) -> RAGState:
        """Increment attempt counter; set a fallback response when giving up."""
        state["attempts"] = state.get("attempts", 0) + 1
        if state["attempts"] >= settings.MAX_RETRIEVAL_ATTEMPTS:
            # All retries exhausted — return a graceful fallback message
            state["response"] = (
                "I wasn't able to find enough relevant information to "
                "answer your question confidently. Please try rephrasing "
                "or asking a more specific question."
            )
        return state

    workflow.add_node("loop_or_end", loop_or_end)

    # ── Conditional edges ─────────────────────────────────────────────────────

    # After analysis: rejected queries terminate immediately
    workflow.add_conditional_edges(
        "analyzer",
        lambda state: "rejected" if state["rejected"] else "to_retriever",
        {"rejected": END, "to_retriever": "retriever"},
    )

    # After retrieval: go to synthesis if we have enough chunks, else check retries
    workflow.add_conditional_edges(
        "retriever",
        lambda state: "synth" if state["enough"] else "loop_or_end",
        {"synth": "synthesizer", "loop_or_end": "loop_or_end"},
    )

    # After loop_or_end: rewrite and retry if attempts remain, otherwise END
    def _loop_logic(state: RAGState) -> str:
        if state["attempts"] < settings.MAX_RETRIEVAL_ATTEMPTS:
            return "rewriter"
        return END

    workflow.add_conditional_edges(
        "loop_or_end",
        _loop_logic,
        {"rewriter": "rewriter", END: END},
    )

    # After rewriting, always go back to the retriever
    workflow.add_edge("rewriter", "retriever")

    workflow.set_entry_point("analyzer")

    return workflow.compile()