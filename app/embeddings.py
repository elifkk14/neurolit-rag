"""Text -> vector. The same model must be used for documents and for questions."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import EMBED_CACHE_DIR, EMBED_MODEL, EMBED_THREADS

# bge models were trained with this prefix on the *question* side only (asymmetric search:
# a short question vs a long passage). Documents are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    # Imported here so that modules which only need the constants do not load ONNX Runtime.
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL, cache_dir=EMBED_CACHE_DIR, threads=EMBED_THREADS)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    # fastembed returns L2-normalized vectors for bge, so cosine similarity = dot product
    return np.array(list(_model().embed(texts, batch_size=batch_size)), dtype=np.float32)


def embed_query(question: str) -> np.ndarray:
    return next(iter(_model().embed([QUERY_PREFIX + question])))
