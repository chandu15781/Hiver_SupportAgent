"""
Main support-agent orchestrator.

SupportAgent is the single entry point for the full pipeline:
    message in → intent classification → retrieval → escalation → reply

All components are injected at construction time so the agent itself
has no hidden state or side effects beyond LLM calls.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import faiss
import numpy as np

from src.config import ProjectConfig, load_config
from src.llm.client import LLMClient
from src.intents.taxonomy import Intent, load_taxonomy
from src.intents.classifier import IntentResult, classify_intent
from src.retrieval.retriever import RetrievedConversation, retrieve
from src.retrieval.index import load_index
from src.agent.escalation import EscalationDecision, decide_escalation
from src.agent.reply_generator import generate_reply

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Complete response from the support agent."""
    message: str  # the original customer message
    intent: IntentResult
    retrieved_conversations: List[RetrievedConversation]
    escalation: EscalationDecision
    draft_reply: Optional[str]  # None if escalated

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "intent": self.intent.to_dict(),
            "retrieved_conversations": [
                rc.to_dict() for rc in self.retrieved_conversations
            ],
            "escalation": self.escalation.to_dict(),
            "draft_reply": self.draft_reply,
        }


class SupportAgent:
    """Orchestrates the full support-agent pipeline."""

    def __init__(
        self,
        llm_client: LLMClient,
        taxonomy: List[Intent],
        index: faiss.Index,
        metadata: List[dict],
        config: Optional[ProjectConfig] = None,
        brand: str = "AppleSupport",
    ):
        self.llm = llm_client
        self.taxonomy = taxonomy
        self.index = index
        self.metadata = metadata
        self.config = config or load_config()
        self.brand = brand

    def handle(self, message: str) -> AgentResponse:
        """Process a customer message through the full pipeline.

        Steps:
            1. Classify intent against the derived taxonomy.
            2. Retrieve similar historical conversations from FAISS.
            3. Decide whether to auto-handle or escalate.
            4. If auto-handling, generate a RAG-grounded draft reply.

        Returns:
            AgentResponse with all intermediate results for transparency.
        """
        logger.info("Processing message: %s…", message[:80])

        # Step 1: Classify intent
        intent_result = classify_intent(
            message=message,
            taxonomy=self.taxonomy,
            llm_client=self.llm,
            brand=self.brand,
        )
        logger.info(
            "Intent: %s (confidence: %.2f)",
            intent_result.intent_name, intent_result.confidence,
        )

        # Step 2: Retrieve similar conversations
        retrieved = retrieve(
            query=message,
            llm_client=self.llm,
            index=self.index,
            metadata=self.metadata,
            top_k=self.config.retrieval.top_k,
            similarity_threshold=0.0,  # return all, let escalation decide
        )

        # Step 3: Escalation decision
        escalation = decide_escalation(
            intent_result=intent_result,
            retrieved_conversations=retrieved,
            intent_threshold=self.config.escalation.intent_threshold,
            retrieval_threshold=self.config.escalation.retrieval_threshold,
        )

        # Step 4: Generate reply (only if auto-handling)
        draft_reply = None
        if escalation.decision == "AUTO_HANDLE":
            draft_reply = generate_reply(
                message=message,
                intent_result=intent_result,
                retrieved_conversations=retrieved,
                llm_client=self.llm,
                brand=self.brand,
            )
        else:
            logger.info("Skipping reply generation — message will be escalated")

        response = AgentResponse(
            message=message,
            intent=intent_result,
            retrieved_conversations=retrieved,
            escalation=escalation,
            draft_reply=draft_reply,
        )

        logger.info(
            "Response: intent=%s, escalation=%s, reply=%s",
            intent_result.intent_name,
            escalation.decision,
            f"{draft_reply[:60]}…" if draft_reply else "N/A",
        )
        return response

    @classmethod
    def from_artifacts(
        cls,
        processed_dir: str = "data/processed",
        config: Optional[ProjectConfig] = None,
    ) -> "SupportAgent":
        """Construct a SupportAgent from saved artifacts on disk.

        Expects:
            <processed_dir>/intent_taxonomy.json
            <processed_dir>/faiss_index.bin
            <processed_dir>/embedding_metadata.json
        """
        from pathlib import Path

        config = config or load_config()
        brand = config.brand.name or "AppleSupport"
        processed = Path(processed_dir)

        llm_client = LLMClient()
        taxonomy = load_taxonomy(str(processed / "intent_taxonomy.json"))
        index = load_index(str(processed / "faiss_index.bin"))

        with open(processed / "embedding_metadata.json") as f:
            metadata = json.load(f)

        # Load the TF-IDF vectorizer so query-time embeddings use the
        # same vocabulary as the indexed documents.
        tfidf_path = processed / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            import pickle
            from src.llm.client import _TfidfEmbedder
            llm_client._tfidf_embedder = _TfidfEmbedder()
            with open(tfidf_path, "rb") as f:
                llm_client._tfidf_embedder._vectorizer = pickle.load(f)
            llm_client._tfidf_embedder._is_fitted = True
            logger.info("Loaded TF-IDF vectorizer from %s", tfidf_path)

        logger.info(
            "Loaded SupportAgent: brand=%s, %d intents, %d indexed conversations",
            brand, len(taxonomy), index.ntotal,
        )
        return cls(
            llm_client=llm_client,
            taxonomy=taxonomy,
            index=index,
            metadata=metadata,
            config=config,
            brand=brand,
        )
