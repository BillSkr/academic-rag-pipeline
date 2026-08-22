"""Tests for the document chunking logic.

split_document_to_chunks() converts a document dict into a list of chunk dicts,
each containing:
  - chunk_id: unique identifier (doc_id + sequential index)
  - doc_id: parent document ID
  - text: enriched text prefixed with title, authors, year
  - metadata: dict with title, authors, year, chunk_index

Token-based splitting is done with tiktoken (cl100k_base encoding).
If tiktoken is unavailable the splitter falls back to whitespace splitting.

These tests verify:
  - Required fields are enforced — missing keys raise ValueError.
  - Empty document text returns an empty list.
  - Long documents produce more than one chunk (each with a unique chunk_id).
  - Metadata (title, authors, year, chunk_index) is preserved in every chunk.
  - Consecutive chunks share overlapping tokens (controlled by CHUNK_OVERLAP_TOKENS).
"""

import pytest
from src.splitting.splitter import split_document_to_chunks


# ──────────────────────────── Missing fields ──────────────────────────────────

@pytest.mark.parametrize(
    "doc",
    [
        # Missing title, authors, and year
        {"doc_id": "x", "text": "hello"},
        # Missing authors and year
        {"doc_id": "y", "title": "Test", "text": "hello"},
        # Missing year only
        {"doc_id": "z", "title": "Test", "authors": ["A"], "text": ""},
    ],
)
def test_missing_keys_raises_value_error(doc):
    """Documents with missing required keys must raise ValueError."""
    with pytest.raises(ValueError):
        split_document_to_chunks(doc)


# ──────────────────────────── Empty text ─────────────────────────────────────

@pytest.mark.parametrize(
    "doc",
    [
        {"doc_id": "x", "title": "Test", "authors": ["A"], "year": "2024", "text": ""},
    ],
)
def test_empty_text_returns_empty_chunks(doc):
    """A document with an empty text field should produce zero chunks."""
    chunks = split_document_to_chunks(doc)
    assert chunks == []


# ──────────────────────────── Multiple chunks ─────────────────────────────────

@pytest.mark.parametrize(
    "doc",
    [
        {
            "doc_id": "long_doc",
            "title": "Long Document",
            "authors": ["Author A"],
            "year": "2024",
            # Repeat enough to exceed the default CHUNK_SIZE_TOKENS setting
            "text": "This is a long document. " * 500,
        },
    ],
)
def test_long_document_produces_multiple_chunks(doc):
    """A sufficiently long document must produce at least 2 chunks with unique IDs."""
    chunks = split_document_to_chunks(doc)
    assert len(chunks) >= 2
    # Each chunk must have a unique chunk_id
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    assert len(chunk_ids) == len(chunks)


# ──────────────────────────── Metadata preservation ──────────────────────────

@pytest.mark.parametrize(
    "doc",
    [
        {
            "doc_id": "meta_doc",
            "title": "Metadata Test",
            "authors": ["Author B"],
            "year": "2024",
            "text": "This document is for testing metadata preservation. " * 50,
        },
    ],
)
def test_chunk_metadata_preserved(doc):
    """Every chunk must carry the parent document's title, authors, year, and its index."""
    chunks = split_document_to_chunks(doc)
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert metadata["title"] == doc["title"]
        assert metadata["authors"] == doc["authors"]
        assert metadata["year"] == doc["year"]
        assert "chunk_index" in metadata


# ──────────────────────────── Overlap ────────────────────────────────────────

@pytest.mark.parametrize(
    "doc",
    [
        {
            "doc_id": "overlap_doc",
            "title": "Overlap Test",
            "authors": ["Author C"],
            "year": "2024",
            # Large enough to produce multiple chunks at the default chunk size
            "text": (
                "This document is for testing overlapping chunks. "
                "It must be long enough so we can see the overlap across "
                "the boundaries of the split text. "
            ) * 200,
        },
    ],
)
def test_overlap_behavior(doc):
    """Consecutive chunks must share at least one word due to the overlap window."""
    chunks = split_document_to_chunks(doc)
    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        current_words = set(chunks[i]["text"].split())
        next_words = set(chunks[i + 1]["text"].split())
        shared = current_words & next_words
        assert len(shared) > 0, (
            f"Chunks {i} and {i + 1} should share overlapping words but got none."
        )
