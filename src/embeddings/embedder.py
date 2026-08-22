"""
Embedding utilities for the Academic RAG Assistant.

Provides OllamaEmbedder — a singleton wrapper around the Ollama embedding
model. Supports single-text embedding and parallel batch embedding.

Configured via settings.OLLAMA_EMBED_MODEL_NAME (default: nomic-embed-text).
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ollama import Client as OllamaClient

from src.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Singleton wrapper around the Ollama embedding endpoint.

    Implemented as a singleton to reuse the same HTTP client connection
    across all nodes that call it during a single graph execution.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        # Only create one instance per process
        if cls._instance is None:
            cls._instance = super(OllamaEmbedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, host: str = None, timeout: int = 30) -> None:
        # Guard: skip re-initialisation on subsequent OllamaEmbedder() calls
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        # Allow overriding via constructor or OLLAMA_BASE_URL env var
        if host is None:
            host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        self.client = OllamaClient(host=host, timeout=timeout)
        logger.info(
            "Initialised OllamaEmbedder with model=%s host=%s",
            settings.OLLAMA_EMBED_MODEL_NAME,
            host,
        )

    def embed(self, text: str, max_retries: int = 3) -> List[float]:
        """Convert a string of text into a dense vector.

        Retries up to `max_retries` times on transient network errors.

        Args:
            text:        The text to embed. Must be non-empty.
            max_retries: Number of retry attempts before raising.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError:   If the input text is empty.
            RuntimeError: If Ollama fails to respond after all retries.
            KeyError:     If the response is missing the 'embedding' key.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug("Embedding attempt %d/%d", attempt, max_retries)
                model_response = self.client.embeddings(
                    model=settings.OLLAMA_EMBED_MODEL_NAME,
                    prompt=text,
                )
            except Exception as exc:
                logger.warning("Embedding attempt %d failed: %s", attempt, exc)
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to get embedding from Ollama after {max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(1)
                continue

            if "embedding" not in model_response:
                raise KeyError("Embedding response missing 'embedding' key.")

            embedding_vector = model_response["embedding"]
            logger.debug("Successfully generated embedding (dim=%d)", len(embedding_vector))
            return embedding_vector

    def embed_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Embed a list of texts concurrently using a thread pool.

        Args:
            texts:       List of non-empty strings to embed.
            max_retries: Retry limit forwarded to each embed() call.

        Returns:
            A list of embedding vectors in the same order as `texts`.
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(
                lambda text: self.embed(text, max_retries=max_retries), texts
            )
            return list(results)
