"""
Central configuration for the Academic RAG Assistant.

All settings are read from environment variables (or a .env file).
Import individual values with:
    from src.config import settings
    settings.CHUNK_SIZE_TOKENS
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Absolute paths resolved relative to this file ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "src" / "vectordb" / "chroma_collection"


class EnvSettings(BaseSettings):
    """Pydantic settings model — values are read from .env or environment."""

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = 800        # max tokens per chunk
    CHUNK_OVERLAP_TOKENS: int = 100     # overlap between consecutive chunks

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K: int = 5                      # number of chunks to retrieve per query
    SIMILARITY_THRESHOLD: float = 0.7  # max cosine distance to accept a chunk
    MAX_RETRIEVAL_ATTEMPTS: int = 3     # retry limit before giving up

    # ── LLM (used by LLMFactory via LiteLLM) ─────────────────────────────────
    MODEL_NAME: str = "ollama/mistral:latest"  # LiteLLM model string
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 2048

    # ── Embedder (Ollama-specific) ────────────────────────────────────────────
    OLLAMA_MODEL_NAME: str = "mistral:latest"
    OLLAMA_EMBED_MODEL_NAME: str = "nomic-embed-text"
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_MAX_TOKENS: int = 2048

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",   # silently ignore unknown env vars
    )


# Instantiate once; re-export as module-level attributes for convenient access
_env = EnvSettings()

CHUNK_SIZE_TOKENS = _env.CHUNK_SIZE_TOKENS
CHUNK_OVERLAP_TOKENS = _env.CHUNK_OVERLAP_TOKENS
TOP_K = _env.TOP_K
SIMILARITY_THRESHOLD = _env.SIMILARITY_THRESHOLD
MAX_RETRIEVAL_ATTEMPTS = _env.MAX_RETRIEVAL_ATTEMPTS
MODEL_NAME = _env.MODEL_NAME
TEMPERATURE = _env.TEMPERATURE
MAX_TOKENS = _env.MAX_TOKENS
OLLAMA_MODEL_NAME = _env.OLLAMA_MODEL_NAME
OLLAMA_EMBED_MODEL_NAME = _env.OLLAMA_EMBED_MODEL_NAME
OLLAMA_TEMPERATURE = _env.OLLAMA_TEMPERATURE
OLLAMA_MAX_TOKENS = _env.OLLAMA_MAX_TOKENS
