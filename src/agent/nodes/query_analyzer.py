"""
Node 1 â€“ Query Analyzer.

Responsibilities:
* Detect whether the incoming query is within the academic domain.
* If the query is complex, decompose it into simpler sub-queries.
* Populate ``state`` accordingly and signal rejection when out-of-scope.
"""

import json

from src.agent.model import LLMFactory
from src.agent.state import RAGState


def analyze_query(state: "RAGState") -> "RAGState":
    """Analyse ``state["user_query"]`` and update the shared state."""
    if not state["user_query"] or not state["user_query"].strip():
        state["rejected"] = True
        state["response"] = "Please enter a valid question."
        return state

    user_query = state["user_query"]
    llm = LLMFactory()

    system_prompt = """You are an academic query classifier. Decide if the following user query is related to academic research.
    Respond with only 'YES' or 'NO'."""
    
    response = llm.generate(system_prompt=system_prompt, user_prompt=user_query)
    response_lower = response.lower()

    if "yes" not in response_lower:
        state["rejected"] = True
        state["response"] = "Sorry, your question does not appear to be related to academic research. Please ask a question about scientific studies, papers, or research topics."
        return state

    system_prompt = """You are a query decomposer. Break down complex academic questions into simpler sub-queries. Return the result strictly as a JSON object with a "sub_queries" key containing a list of strings. Do not include markdown formatting or explanations.
    For example:
    {"sub_queries": ["What is the role of SOD1 protein in ALS?", "What are the known mutations in SOD1 that cause ALS?"]}"""
    
    response = llm.generate(system_prompt=system_prompt, user_prompt=user_query)
    
    state["rejected"] = False

    try:
        clean_response = response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_response)
        state["sub_queries"] = parsed.get("sub_queries", [])
    except Exception as e:
        print(f"Failed to parse sub-queries JSON: {e}")
        state["sub_queries"] = []

    state["current_query"] = state["user_query"]
    return state