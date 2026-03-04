"""Config package - Global settings và configurations."""
from .settings import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    CHROMA_PATH,
    JSON_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_INITIAL,
    TOP_K_FINAL,
    WEB_SEARCH_COOLDOWN,
    MAX_WEB_SNIPPETS,
    MIN_WEB_RESULT_LENGTH,
    DEVICE,
)

__all__ = [
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "CHROMA_PATH",
    "JSON_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K_INITIAL",
    "TOP_K_FINAL",
    "WEB_SEARCH_COOLDOWN",
    "MAX_WEB_SNIPPETS",
    "MIN_WEB_RESULT_LENGTH",
    "DEVICE",
]
