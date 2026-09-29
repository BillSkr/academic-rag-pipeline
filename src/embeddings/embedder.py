"""
Embedding utilities for the Academic RAG Assistant.

Provides LocalEmbedder — a singleton wrapper around sentence-transformers.
Supports single-text embedding and parallel batch embedding.

Configured via settings.EMBED_MODEL_NAME (default: all-MiniLM-L6-v2).
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Singleton wrapper around a local sentence-transformers model.

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

        model_name = getattr(settings, "EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
        logger.info(f"Loading local embedding model: {model_name}")
        # Load the model explicitly on CPU to ensure it works anywhere
        self.model = SentenceTransformer(model_name, device='cpu')
        logger.info("Local embedding model loaded successfully.")

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

        embedding_vector = self.model.encode(text, show_progress_bar=False)
        return embedding_vector.tolist()

    def embed_batch(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts:       List of non-empty strings to embed.
            max_retries: Ignored.

        Returns:
            A list of embedding vectors in the same order as `texts`.
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

