"""Ingestion package - Document loading và preparation."""
from .pdf_loader import (
    get_embeddings,
    process_and_index_documents,
    load_vector_db,
)

__all__ = [
    "get_embeddings",
    "process_and_index_documents",
    "load_vector_db",
]
