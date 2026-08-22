"""
Document ingestion utilities.

Loads academic documents from two possible sources:
  1. data/corpus.json — structured JSON produced by fetch_papers.py (preferred).
  2. data/academic_papers/ — raw PDF or plain-text files (fallback).

The returned list of dicts always has the keys:
    doc_id, title, authors, year, text
"""

import json
import logging
from pathlib import Path

import pymupdf

from src.config import settings

logger = logging.getLogger(__name__)


def load_academic_documents(source_dir: Path = None) -> list[dict]:
    """Load academic documents, preferring corpus.json over raw files.

    Args:
        source_dir: Override for the directory scanned for PDF/TXT files.
                    Defaults to data/academic_papers/ if not provided.

    Returns:
        List of document dicts with keys: doc_id, title, authors, year, text.
    """
    corpus_path = settings.DATA_DIR / "corpus.json"

    # Prefer the structured JSON corpus when it exists
    if corpus_path.exists():
        logger.info("Loading documents from %s", corpus_path)
        return _load_from_corpus(corpus_path)

    # Fall back to scanning the papers directory for PDF / TXT files
    if source_dir is None:
        source_dir = settings.DATA_DIR / "academic_papers"
    else:
        source_dir = Path(source_dir)

    logger.info("corpus.json not found — scanning %s for documents", source_dir)
    documents = []
    for file_path in source_dir.glob("**/*"):
        if file_path.suffix.lower() in {".pdf", ".txt"}:
            doc = parse_document(file_path)
            if doc:
                documents.append(doc)

    return documents


def _load_from_corpus(corpus_path: Path) -> list[dict]:
    """Read documents from the JSON corpus produced by fetch_papers.py.

    The 'authors' field in corpus.json is a list; it is joined into a string
    here so every document has a consistent format.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    documents = []
    for record in records:
        # Combine title and abstract into a single searchable text field
        text = f"Title: {record.get('title', '')}\n\nAbstract: {record.get('abstract', '')}"
        documents.append({
            "doc_id": str(record.get("pmid", "unknown")),
            "title": record.get("title", ""),
            "authors": ", ".join(record.get("authors", [])),
            "year": record.get("year", ""),
            "text": text,
        })

    return documents


def parse_document(file_path: Path) -> dict | None:
    """Parse a single PDF or plain-text file into a document dict.

    Returns None and logs a warning if the file cannot be read.
    Note: PDF-sourced documents will have empty authors and year fields
    since that metadata is not embedded in raw files.
    """
    try:
        if file_path.suffix.lower() == ".pdf":
            # Extract text from all pages using PyMuPDF
            with pymupdf.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
        return None

    return {
        "doc_id": file_path.stem,
        "title": file_path.stem,    # filename used as title for raw files
        "authors": "",              # not available from raw files
        "year": "",                 # not available from raw files
        "text": text,
    }
