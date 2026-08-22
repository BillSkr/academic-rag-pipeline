"""Chunking logic."""

from typing import List

from src.config import settings

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _encode(text: str) -> List[int]:
        return _ENCODING.encode(text)

    def _decode(tokens: List[int]) -> str:
        return _ENCODING.decode(tokens)

except Exception:
    def _encode(text: str) -> List[str]:
        return text.split()

    def _decode(tokens: List[str]) -> str:
        return " ".join(tokens)


_MIN_CHUNK_SIZE = max(1, settings.CHUNK_SIZE_TOKENS // 10)


def split_document_to_chunks(document: dict) -> List[dict]:
    """Split a single document into token-based chunks."""
    required_keys = {"doc_id", "title", "authors", "year", "text"}
    missing = required_keys - document.keys()
    if missing:
        raise ValueError(f"Document dict is missing required keys: {missing}")

    raw_text = document["text"]
    if not raw_text or not raw_text.strip():
        return []

    tokens = _encode(raw_text)
    i = 0
    chunks: List[dict] = []
    step = settings.CHUNK_SIZE_TOKENS - settings.CHUNK_OVERLAP_TOKENS

    while i < len(tokens):
        window = tokens[i : i + settings.CHUNK_SIZE_TOKENS]

        if len(window) < _MIN_CHUNK_SIZE and i > 0:
            break

        chunk_text = _decode(window)
        enriched_text = (
            f"Title: {document.get('title', 'Unknown')}\n"
            f"Authors: {document.get('authors', 'Unknown')}\n"
            f"Year: {document.get('year', 'Unknown')}\n\n"
            f"{chunk_text}"
        )

        chunk = {
            "chunk_id": f"{document['doc_id']}_chunk_{len(chunks)}",
            "doc_id": document["doc_id"],
            "text": enriched_text,
            "metadata": {
                "title": document["title"],
                "authors": document["authors"],
                "year": document["year"],
                "chunk_index": len(chunks),
            },
        }
        chunks.append(chunk)
        i += step

    return chunks

