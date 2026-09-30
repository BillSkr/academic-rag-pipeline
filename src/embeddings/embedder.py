"""
Embedding utilities for the Academic RAG Assistant.

Provides LocalEmbedder — a singleton wrapper around chromadb's DefaultEmbeddingFunction (ONNX).
Supports single-text embedding and parallel batch embedding.

Configured via settings.EMBED_MODEL_NAME (default: all-MiniLM-L6-v2).
"""

import logging

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Singleton wrapper around a lightweight local ONNX embedding model.

    Implemented as a singleton to avoid reloading the model into memory
    multiple times across different modules.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        # Only create one instance per process
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Guard: skip re-initialisation on subsequent LocalEmbedder() calls
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        logger.info("Loading local embedding model: all-MiniLM-L6-v2 (ONNX)")
        # ChromaDB's default embedding function uses ONNX and takes very little memory
        self.ef = DefaultEmbeddingFunction()
        logger.info("Local ONNX embedding model loaded successfully.")

    def embed(self, text: str, max_retries: int = 3) -> list[float]:
        """Convert a string of text into a dense vector.

        Args:
            text:        The text to embed. Must be non-empty.
            max_retries: Ignored for local embeddings, kept for API compatibility.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # DefaultEmbeddingFunction returns a list of embeddings
        embedding_vector = self.ef([text])[0]
        return embedding_vector

    def embed_batch(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts:       List of non-empty strings to embed.
            max_retries: Ignored.

        Returns:
            A list of embedding vectors in the same order as `texts`.
        """
        return self.ef(texts)
