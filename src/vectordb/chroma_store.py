"""
ChromaDB vector-store wrapper for the Academic RAG Assistant.

ChromaVectorStore provides:
  - upsert_documents(): batch-insert chunks with embeddings and metadata.
  - similarity_search(): hybrid dense + keyword retrieval with:
      * Automatic year-based metadata filtering from query text.
      * Deduplication of dense and keyword results.
      * Parent-child retrieval (fetches the full parent chunk when a child chunk matches).
      * Automatic retry (3 attempts) on transient ChromaDB errors.

The collection is persisted to disk at settings.CHROMA_PERSIST_DIR.
"""

import logging
import re

import chromadb

from src.config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Singleton high-level API around a single persistent Chroma collection."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        # Reuse the same client and collection across all callers
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Skip re-initialisation on repeated ChromaVectorStore() calls
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        # Open (or create) the persistent collection on disk
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(name="academic_docs")

    def upsert_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        """Persist a batch of documents with their embeddings and metadata.

        Uses upsert so re-running the pipeline won't create duplicates.
        """
        self.collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = settings.TOP_K,
        metadata_filter: dict = None,
        query_text: str = None,
    ) -> list[dict]:
        """Retrieve the top-k most similar chunks using hybrid retrieval.

        Steps:
          1. Parse a year constraint from ``query_text`` and add it to
             ``metadata_filter`` automatically (e.g. "since 2020").
          2. Run dense vector search via the query embedding.
          3. Optionally run a keyword search when ``query_text`` is provided,
             and merge results with the dense hits (deduplication by ID).
          4. Sort all unique results by distance (ascending).
          5. Apply parent-child retrieval: if a chunk has a ``parent_id``
             in its metadata, fetch the full parent document instead.
          6. Return the top-k deduplicated results.

        Args:
            query_embedding: Dense vector for the query.
            k:               Maximum number of results to return.
            metadata_filter: Optional Chroma ``where`` filter dict.
            query_text:      Raw query string for keyword search and year parsing.

        Returns:
            List of result dicts with keys: id, document, metadata, distance.

        Raises:
            RuntimeError: After 3 failed attempts on transient errors.
        """
        metadata_filter = metadata_filter or {}

        # ── Step 1: Auto-parse year constraint from query text ────────────────
        if query_text:
            year_match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})\b', query_text)
            if year_match:
                year = int(year_match.group(1))
                if "since" in query_text.lower() or "after" in query_text.lower():
                    metadata_filter["year"] = {"$gte": year}
                else:
                    metadata_filter["year"] = year

        for attempt in range(3):
            try:
                # ── Step 2: Dense vector search ───────────────────────────────
                query_kwargs = {
                    "query_embeddings": [query_embedding],
                    "n_results": k,
                    "include": ["documents", "metadatas", "distances"],
                }
                if metadata_filter:
                    query_kwargs["where"] = metadata_filter

                results = self.collection.query(**query_kwargs)

                # Build a dict keyed by chunk ID to enable deduplication
                combined = {
                    results["ids"][0][j]: {
                        "id": results["ids"][0][j],
                        "document": results["documents"][0][j],
                        "metadata": results["metadatas"][0][j],
                        "distance": results["distances"][0][j],
                    }
                    for j in range(len(results["ids"][0]))
                }

                # ── Step 3: Keyword search (optional) ────────────────────────
                if query_text:
                    text_kwargs = {
                        "query_texts": [query_text],
                        "n_results": k,
                        "include": ["documents", "metadatas", "distances"],
                    }
                    if metadata_filter:
                        text_kwargs["where"] = metadata_filter

                    text_results = self.collection.query(**text_kwargs)
                    for j in range(len(text_results["ids"][0])):
                        doc_id = text_results["ids"][0][j]
                        # Only add if not already present from dense search
                        if doc_id not in combined:
                            combined[doc_id] = {
                                "id": doc_id,
                                "document": text_results["documents"][0][j],
                                "metadata": text_results["metadatas"][0][j],
                                "distance": text_results["distances"][0][j],
                            }

                # ── Step 4: Sort merged results by similarity ─────────────────
                sorted_results = sorted(combined.values(), key=lambda x: x["distance"])

                # ── Step 5 & 6: Parent-child retrieval + deduplication ─────────
                final_top_k = []
                seen_ids: set = set()

                for res in sorted_results:
                    metadata = res["metadata"] or {}
                    parent_id = metadata.get("parent_id")

                    if parent_id:
                        # This is a child chunk — fetch its parent document instead
                        if parent_id in seen_ids:
                            continue   # already included via another child
                        seen_ids.add(parent_id)
                        parent_data = self.collection.get(ids=[parent_id])
                        if (
                            parent_data
                            and parent_data.get("documents")
                            and len(parent_data["documents"]) > 0
                        ):
                            res["document"] = parent_data["documents"][0]
                            if parent_data.get("metadatas"):
                                res["metadata"] = parent_data["metadatas"][0]
                    else:
                        if res["id"] in seen_ids:
                            continue
                        seen_ids.add(res["id"])

                    final_top_k.append(res)
                    if len(final_top_k) >= k:
                        break   # collected enough results

                return final_top_k

            except Exception as exc:
                logger.warning("ChromaDB query attempt %d/3 failed: %s", attempt + 1, exc)
                if attempt == 2:
                    raise RuntimeError("ChromaDB query failed after 3 attempts") from exc

        return []   # unreachable but satisfies type checkers