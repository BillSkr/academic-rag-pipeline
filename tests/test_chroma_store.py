"""Tests for the ChromaDB vector store wrapper.

The ChromaVectorStore class wraps a chromadb collection and provides:
  - upsert_documents(): batch-insert chunks with their embeddings.
  - similarity_search(): retrieve the top-k most similar chunks with retry logic.

These tests verify:
  - Documents are upserted correctly and retrievable.
  - similarity_search formats results into the expected dict structure.
  - Transient errors (RuntimeError) trigger automatic retries.
  - Persistent failures raise a RuntimeError after exhausting all retries.

The underlying chromadb collection is replaced with a lightweight MockCollection
so no real database is started.
"""

import pytest
from src.vectordb.chroma_store import ChromaVectorStore


# ──────────────────────────── MockCollection ─────────────────────────────────

class MockCollection:
    """Minimal stand-in for a chromadb Collection object.

    Tracks how many times .query() has been called and raises a RuntimeError
    for the first `fail_count` calls to simulate transient failures.
    """

    def __init__(self):
        self.call_count = 0   # number of times query() has been called
        self.fail_count = 0   # number of leading calls that should raise

    def upsert(self, ids, embeddings, metadatas, documents):
        """Record the args so tests can inspect what was passed."""
        self.upsert_called_with = (ids, embeddings, metadatas, documents)

    def query(self, query_embeddings, n_results, include, where=None, query_texts=None):
        """Return a fixed result dict, raising on the first `fail_count` calls."""
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise RuntimeError("Mock transient ChromaDB error")

        return {
            "ids": [["test_doc_1", "test_doc_2"]],
            "documents": [["Sample text 1", "Sample text 2"]],
            "metadatas": [[{"title": "Test 1"}, {"title": "Test 2"}]],
            "distances": [[0.1, 0.2]],
        }


# ──────────────────────────── Fixtures ───────────────────────────────────────

@pytest.fixture
def mock_chroma_store():
    """Return a (ChromaVectorStore, MockCollection) pair.

    The store's internal .collection is replaced with the mock so no real
    ChromaDB client or persistence layer is involved.
    """
    store = ChromaVectorStore()
    mock_collection = MockCollection()
    store.collection = mock_collection
    return store, mock_collection


# ──────────────────────────── Tests ──────────────────────────────────────────

def test_add_and_retrieve(mock_chroma_store):
    """Documents can be upserted and then retrieved via similarity_search."""
    store, mock_collection = mock_chroma_store

    # Insert two fake documents
    store.upsert_documents(
        ids=["id1", "id2"],
        embeddings=[[0.1] * 768, [0.2] * 768],
        metadatas=[{"title": "Doc 1"}, {"title": "Doc 2"}],
        documents=["text 1", "text 2"],
    )
    # Verify the IDs were forwarded to the underlying collection
    assert mock_collection.upsert_called_with[0] == ["id1", "id2"]

    # Query and check the result format
    results = store.similarity_search(query_embedding=[0.1] * 768, k=2)
    assert len(results) == 2
    assert results[0]["id"] == "test_doc_1"
    assert results[0]["document"] == "Sample text 1"
    assert results[0]["metadata"]["title"] == "Test 1"
    assert results[0]["distance"] == 0.1


def test_retry_transient_failure(mock_chroma_store):
    """similarity_search retries on transient errors and succeeds on the 3rd attempt."""
    store, mock_collection = mock_chroma_store
    mock_collection.fail_count = 2  # fail the first two calls

    results = store.similarity_search(query_embedding=[0.1] * 768, k=2)

    assert len(results) == 2
    # Total calls should be 3: 2 failures + 1 success
    assert mock_collection.call_count == 3


def test_persistent_failure(mock_chroma_store):
    """similarity_search raises RuntimeError after exhausting all retries."""
    store, mock_collection = mock_chroma_store
    mock_collection.fail_count = 5  # more failures than the retry limit

    with pytest.raises(RuntimeError, match="ChromaDB query failed after 3 attempts"):
        store.similarity_search(query_embedding=[0.1] * 768, k=2)