"""Text -> vector. The same model must be used for documents and for questions."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import EMBED_DEVICE, EMBED_MODEL

# bge models were trained with this prefix on the *question* side only (asymmetric search:
# a short question vs a long passage). Documents are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    # Imported here so that modules which only need the constants do not load torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    # normalize_embeddings=True -> every vector has length 1, so cosine similarity = dot product
    return _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 500
    )


def embed_query(question: str) -> np.ndarray:
    return _model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
