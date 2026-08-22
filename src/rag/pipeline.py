"""
High-level RAG pipeline orchestrator.

build_vector_store() is the single entry point for indexing:
  1. Load raw academic documents from corpus.json or the papers directory.
  2. Split each document into token-sized chunks.
  3. Embed all chunks in parallel via OllamaEmbedder.
  4. Upsert chunks and embeddings into the persistent ChromaDB collection.

Run once before starting the API server, or again whenever new papers are added.
"""

import logging

from tqdm import tqdm

from src.config import settings
from src.ingestion import loader
from src.splitting import splitter
from src.embeddings.embedder import OllamaEmbedder
from src.vectordb.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def build_vector_store() -> None:
    """Populate the persistent Chroma collection from source documents.

    Raises:
        RuntimeError: If document loading, embedding, or persistence fails.
    """

    # ── 1. Load documents ─────────────────────────────────────────────────────
    try:
        docs = loader.load_academic_documents()
    except Exception as exc:
        raise RuntimeError(f"Failed to load academic documents: {exc}") from exc

    if not docs:
        logger.warning("No documents found in the data directory.")
        return

    # ── 2. Split into chunks ──────────────────────────────────────────────────
    all_chunks = []
    for doc in docs:
        try:
            all_chunks.extend(splitter.split_document_to_chunks(doc))
        except Exception as exc:
            # Log and skip malformed documents rather than aborting the whole run
            logger.warning("Skipping document %s: %s", doc.get("doc_id", "?"), exc)

    if not all_chunks:
        logger.warning("No chunks produced from the loaded documents.")
        return

    # ── 3. Embed chunks in parallel ───────────────────────────────────────────
    embedder = OllamaEmbedder()
    ids = [chunk["chunk_id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    texts = [chunk["text"] for chunk in all_chunks]

    logger.info("Embedding %d chunks in parallel...", len(texts))
    print(f"Embedding {len(texts)} chunks in parallel...")
    try:
        embeddings = embedder.embed_batch(texts)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    # ── 4. Persist to ChromaDB ────────────────────────────────────────────────
    try:
        store = ChromaVectorStore()
        store.upsert_documents(ids, embeddings, metadatas, texts)
        logger.info("Successfully indexed %d chunks into ChromaDB.", len(ids))
        print(f"Successfully indexed {len(ids)} chunks into ChromaDB.")
    except Exception as exc:
        raise RuntimeError(f"Failed to persist embeddings to ChromaDB: {exc}") from exc