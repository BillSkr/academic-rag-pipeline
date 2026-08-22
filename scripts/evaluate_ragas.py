"""
Offline RAGAS Evaluation Script.

Run after building the vector store to measure pipeline quality
using four RAGAS metrics against a synthetically generated testset:

    - context_precision : are the retrieved chunks relevant?
    - context_recall    : are all relevant chunks retrieved?
    - faithfulness      : does the answer stick to the provided context?
    - answer_relevancy  : is the answer relevant to the question?

Usage:
    python scripts/evaluate_ragas.py

Requirements:
    - Ollama must be running locally with the configured model.
    - corpus.json must exist in the data/ directory.
    - ragas, langchain-ollama, datasets must be installed.
"""

import json
import os
import sys
import types

# ── Add project root to sys.path so 'src' is importable ──────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Stub out the optional VertexAI backend to avoid ImportError ───────────────
# langchain-community sometimes imports this even when not used.
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = fake_vertexai

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.testset.generator import TestsetGenerator
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import settings
from src.ingestion.loader import load_academic_documents


def main() -> None:
    """Run the full RAGAS evaluation pipeline."""
    print("Running RAGAS evaluation...")

    # ── 1. Load source documents ──────────────────────────────────────────────
    print("Loading documents for testset generation...")
    raw_docs = load_academic_documents()
    documents = [
        Document(
            page_content=d["text"],
            metadata={"title": d["title"], "authors": d["authors"], "year": d["year"]},
        )
        for d in raw_docs
        if d.get("text")
    ]

    # ── 2. Configure local Ollama LLM and embeddings for RAGAS ───────────────
    # Strip the "ollama/" prefix that LiteLLM uses; langchain-ollama doesn't need it
    model_name = getattr(settings, "MODEL_NAME", "mistral:latest").replace("ollama/", "")
    ragas_llm = ChatOllama(model=model_name, base_url="http://localhost:11434")
    ragas_embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL_NAME, base_url="http://localhost:11434"
    )

    # ── 3. Generate a synthetic testset from the loaded documents ─────────────
    print("Initialising TestsetGenerator...")
    generator = TestsetGenerator.from_langchain(
        generator_llm=ragas_llm,
        critic_llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("Generating synthetic testset (this may take a while)...")
    testset = generator.generate_with_langchain_docs(documents, test_size=3)
    dataset = testset.to_dataset()

    # ── 4. Run RAGAS metrics ──────────────────────────────────────────────────
    print("Evaluating metrics with RAGAS (this may take a while)...")
    evaluation_result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(evaluation_result)

    # ── 5. Persist results to disk ────────────────────────────────────────────
    output_path = "ragas_results.json"
    with open(output_path, "w") as f:
        json.dump(evaluation_result.to_pandas().to_dict(orient="records"), f, indent=4)
    print(f"Detailed results saved to {output_path}")

    # ── 6. CI/CD gate: fail if context_precision falls below baseline ─────────
    baseline = 0.8
    if "context_precision" in evaluation_result:
        score = evaluation_result["context_precision"]
        assert score >= baseline, (
            f"Evaluation Failed: context_precision ({score:.3f}) "
            f"is below the baseline ({baseline})."
        )
        print(f"CI/CD Check Passed: context_precision ({score:.3f}) >= {baseline}")


if __name__ == "__main__":
    main()
