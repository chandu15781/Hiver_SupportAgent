"""
Retrieve similar historical conversations for a new customer message.

At query time:
1. Embed the incoming customer message.
2. Search the FAISS index for the top-k most similar conversations.
3. Filter by similarity threshold.
4. Return the matching conversations with similarity scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import List

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedConversation:
    """A historical conversation retrieved as context for RAG."""
    conversation_id: str
    brand: str
    similarity: float
    query_text: str  # the customer-side text that was embedded
    messages: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def retrieve(
    query: str,
    llm_client,
    index: faiss.Index,
    metadata: List[dict],
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> List[RetrievedConversation]:
    """Find the most similar historical conversations for a query.

    Args:
        query: the customer message to find similar conversations for.
        llm_client: an LLMClient instance for embedding the query.
        index: a FAISS index built from conversation embeddings.
        metadata: list of dicts aligned with the index (same order).
        top_k: maximum number of results to return.
        similarity_threshold: minimum cosine similarity to include.

    Returns:
        List of RetrievedConversation objects, sorted by similarity descending.
    """
    # Embed the query
    query_embedding = np.array(
        llm_client.embed([query]), dtype=np.float32
    )
    faiss.normalize_L2(query_embedding)

    # Search — request more than top_k so we can filter by threshold
    search_k = min(top_k * 2, index.ntotal)
    similarities, indices = index.search(query_embedding, search_k)

    results: List[RetrievedConversation] = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:  # FAISS sentinel for "no result"
            continue
        if sim < similarity_threshold:
            continue

        meta = metadata[idx]
        results.append(
            RetrievedConversation(
                conversation_id=meta["conversation_id"],
                brand=meta["brand"],
                similarity=float(sim),
                query_text=meta["query_text"],
                messages=meta["messages"],
            )
        )
        if len(results) >= top_k:
            break

    logger.info(
        "Retrieved %d conversations (threshold=%.2f, top_k=%d)",
        len(results), similarity_threshold, top_k,
    )
    return results
