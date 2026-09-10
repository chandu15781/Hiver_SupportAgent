"""
FAISS index management for conversation retrieval.

Uses a simple flat-L2 index — the dataset is small enough (~tens to
low thousands of conversations) that approximate search adds complexity
without meaningful speed benefit. If the corpus grows past ~100k
conversations, switch to IndexIVFFlat.
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS flat inner-product index from an embedding matrix.

    We normalize embeddings and use IndexFlatIP so that similarity scores
    are cosine similarities (range [0, 1] for normalized vectors),
    which is more interpretable for threshold-based filtering than L2.

    Args:
        embeddings: numpy array of shape (n, dim), float32.

    Returns:
        A trained FAISS index ready for search.
    """
    embeddings = embeddings.astype(np.float32)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(
        "Built FAISS index: %d vectors, %d dimensions", index.ntotal, dim
    )
    return index


def save_index(index: faiss.Index, path: str) -> None:
    """Persist a FAISS index to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    logger.info("Saved FAISS index to %s (%d vectors)", out, index.ntotal)


def load_index(path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    index = faiss.read_index(str(path))
    logger.info("Loaded FAISS index from %s (%d vectors)", path, index.ntotal)
    return index
