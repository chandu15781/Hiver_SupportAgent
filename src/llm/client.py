"""
Thin wrapper around LLM providers for chat completions and embeddings.

Supports:
- **Groq** (default): fast inference via OpenAI-compatible API
- **OpenAI**: standard OpenAI API

For embeddings, uses scikit-learn TF-IDF locally since Groq does not
offer an embedding endpoint. This avoids any external API cost for
embeddings and works well for the small dataset sizes in this project.

Requires GROQ_API_KEY (or OPENAI_API_KEY) in the environment.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

load_dotenv()

logger = logging.getLogger(__name__)

# Retry settings
_MAX_RETRIES = 5
_BASE_DELAY = 1.0  # seconds
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)

# Provider configs
_PROVIDER_CONFIGS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "qwen/qwen3.8-27b",
    },
    "openai": {
        "base_url": None,  # uses default
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}


@dataclass
class UsageStats:
    """Accumulates token usage across calls for cost visibility."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    total_calls: int = 0

    def log_summary(self) -> str:
        return (
            f"LLM usage — calls: {self.total_calls}, "
            f"prompt_tok: {self.prompt_tokens}, "
            f"completion_tok: {self.completion_tokens}, "
            f"embed_tok: {self.embedding_tokens}"
        )


class LLMClient:
    """Unified client for chat completions and local TF-IDF embeddings."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._provider = provider or os.getenv("LLM_PROVIDER", "groq")
        config = _PROVIDER_CONFIGS.get(self._provider, _PROVIDER_CONFIGS["groq"])

        self._api_key = api_key or os.getenv(config["api_key_env"], "")
        if not self._api_key:
            raise ValueError(
                f"{config['api_key_env']} is not set. Add it to your .env file or "
                f"pass it explicitly to LLMClient."
            )

        self.chat_model = chat_model or os.getenv("LLM_MODEL", config["default_model"])
        self._base_url = base_url or config["base_url"]

        client_kwargs = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        self._client = OpenAI(**client_kwargs)
        self.usage = UsageStats()

        # Lazy-initialized TF-IDF embedder
        self._tfidf_embedder: Optional[_TfidfEmbedder] = None

        logger.info(
            "LLMClient initialized: provider=%s, model=%s, base_url=%s",
            self._provider, self.chat_model, self._base_url or "default",
        )

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat-completion request and return the assistant content.

        ``messages`` follows the OpenAI format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        kwargs: dict = dict(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self._retry(lambda: self._client.chat.completions.create(**kwargs))

        self.usage.total_calls += 1
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens
            self.usage.completion_tokens += response.usage.completion_tokens

        content = response.choices[0].message.content or ""
        logger.debug("Chat response (%d chars): %s…", len(content), content[:120])
        return content

    # ------------------------------------------------------------------
    # Embeddings (local TF-IDF — no API call needed)
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts using local TF-IDF.

        Uses scikit-learn TF-IDF vectorization instead of an API-based
        embedding model. This is fast, free, and sufficient for the
        small dataset sizes in this project.
        """
        if not texts:
            return []

        if self._tfidf_embedder is None:
            self._tfidf_embedder = _TfidfEmbedder()

        embeddings = self._tfidf_embedder.embed(texts)
        self.usage.total_calls += 1
        logger.debug("Embedded %d texts locally (TF-IDF)", len(texts))
        return embeddings

    def fit_embedder(self, corpus: list[str]) -> None:
        """Fit the TF-IDF embedder on a corpus before embedding queries.

        Call this once with all the documents that will be indexed,
        so that query-time embeddings use the same vocabulary.
        """
        self._tfidf_embedder = _TfidfEmbedder()
        self._tfidf_embedder.fit(corpus)
        logger.info("TF-IDF embedder fitted on %d documents", len(corpus))

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------
    @staticmethod
    def _retry(fn, max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY):
        """Exponential backoff for transient API errors."""
        for attempt in range(max_retries):
            try:
                return fn()
            except _RETRYABLE as exc:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d): %s — waiting %.1fs",
                    attempt + 1, max_retries, exc, delay,
                )
                time.sleep(delay)


# ------------------------------------------------------------------
# Local TF-IDF Embedder
# ------------------------------------------------------------------

class _TfidfEmbedder:
    """Wraps scikit-learn TF-IDF for local embedding without API calls.

    The embedder maintains a fitted vocabulary so that both indexed
    documents and runtime queries map to the same feature space.
    """

    def __init__(self, max_features: int = 512):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit the vectorizer on a corpus."""
        self._vectorizer.fit(texts)
        self._is_fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Transform texts into TF-IDF vectors.

        If the vectorizer hasn't been fitted yet, fit_transform is used
        (suitable for the indexing pass). Subsequent calls use transform
        only (suitable for query-time embedding).
        """
        if not self._is_fitted:
            matrix = self._vectorizer.fit_transform(texts)
            self._is_fitted = True
        else:
            matrix = self._vectorizer.transform(texts)

        return matrix.toarray().tolist()
