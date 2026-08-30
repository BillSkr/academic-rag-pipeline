"""
Node 3 â€“ Synthesizer.

* Receives validated chunks from the Retriever node.
* Generates a natural-language answer that includes inline citations.
* Restricts the LLM to use only the provided snippets.
"""

from src.agent.model import LLMFactory
from src.agent.state import RAGState


def synthesize_answer(state: "RAGState") -> "RAGState":
    """Produce the final answer with citations."""
    model = LLMFactory()

    chunks_text = "\n".join(c["document"] for c in state["retrieved_chunks"])
    citations_meta = "\n".join(
        f"[Document: {c['metadata'].get('title', 'Unknown')}, {c['metadata'].get('year', 'N/A')}]"
        for c in state["retrieved_chunks"]
    )

    history_text = ""
    if state.get("chat_history"):
        history_text = "Conversation History:\n"
        for msg in state["chat_history"]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_text += f"{role}: {msg.get('content')}\n"
        history_text += "\n"

    prompt = (
        f"{history_text}"
        f"Question: {state['current_query']}\n\n"
        f"Excerpts:\n{chunks_text}\n\n"
        f"Citation guide:\n{citations_meta}\n\n"
        "Provide a concise, well-cited answer."
    )

    answer = model.generate(
        system_prompt=(
            "You are an academic research assistant. Answer the user's question using ONLY "
            "the provided excerpts. Cite each fact using the format [Document: <title>, <year>]. "
            "Do not hallucinate or use outside knowledge."
        ),
        user_prompt=prompt,
    )
    state["response"] = answer
    state["citations"] = state["retrieved_chunks"]
    return state