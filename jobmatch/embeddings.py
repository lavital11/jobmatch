"""Shared sentence-transformers model loading for JobMatch.

Both the ingestion script (embeds job postings with the "passage: " prefix)
and the retrieval module (embeds CVs with the "query: " prefix) need the same
underlying model. Loading it is the expensive part (weights download + load
onto device), so it's cached here and shared.
"""

from __future__ import annotations

from functools import lru_cache

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the e5 model once per process and reuse it on every subsequent call."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode already-prefixed ("passage: "/"query: ") texts into normalized vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()
