"""
Embed conversations for retrieval.

Each conversation is represented as a single embedding by concatenating
its customer messages (the "query surface" — what a new customer message
will be compared against).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _conversation_to_text(conv: dict) -> str:
    """Create a single text string from a conversation's customer messages.

    We embed the customer side only because at retrieval time we're matching
    a new customer message against historical customer messages to find
    "similar issues", not matching against agent replies.
    """
    customer_msgs = [
        msg["text"] for msg in conv["messages"] if msg["role"] == "customer"
    ]
    return " ".join(customer_msgs)


def embed_conversations(
    conversations: List[dict],
    llm_client,
    out_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Embed all conversations and return (embeddings_matrix, metadata).

    Args:
        conversations: list of conversation dicts.
        llm_client: an LLMClient instance (must support .embed()).
        out_dir: optional directory to save embeddings and metadata.

    Returns:
        embeddings: numpy array of shape (n_conversations, embedding_dim)
        metadata: list of dicts with conversation_id, brand, text, messages
    """
    texts = []
    metadata = []
    for conv in conversations:
        text = _conversation_to_text(conv)
        if not text.strip():
            continue
        texts.append(text)
        metadata.append({
            "conversation_id": conv["conversation_id"],
            "brand": conv["brand"],
            "query_text": text,
            "messages": conv["messages"],
        })

    logger.info("Embedding %d conversations...", len(texts))

    # Fit the embedder on the corpus so query-time embeddings share
    # the same vocabulary (important for TF-IDF local embeddings).
    if hasattr(llm_client, "fit_embedder"):
        llm_client.fit_embedder(texts)

    raw_embeddings = llm_client.embed(texts)
    embeddings = np.array(raw_embeddings, dtype=np.float32)
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "embeddings.npy", embeddings)
        with open(out_path / "embedding_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save the TF-IDF vectorizer for query-time embedding
        if hasattr(llm_client, "_tfidf_embedder") and llm_client._tfidf_embedder:
            import pickle
            with open(out_path / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(llm_client._tfidf_embedder._vectorizer, f)
            logger.info("Saved TF-IDF vectorizer to %s", out_path / "tfidf_vectorizer.pkl")

        logger.info("Saved embeddings to %s", out_path)

    return embeddings, metadata
