"""
Build a golden evaluation set from historical conversations.

Takes real conversations and creates test cases by:
1. Using the customer's first message as the test input.
2. Using the actual agent reply as the reference reply.
3. Using the LLM to assign the "expected" intent from the taxonomy
   (since we don't have ground-truth intent labels in the raw data).

This gives us a test set grounded in real data rather than synthetic
examples, though the intent labels are LLM-assigned (acknowledged
limitation).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from src.intents.taxonomy import Intent
from src.intents.classifier import classify_intent, IntentResult

logger = logging.getLogger(__name__)


@dataclass
class GoldenExample:
    """A single evaluation example."""
    conversation_id: str
    customer_message: str
    expected_intent: str
    intent_confidence: float
    reference_reply: str  # the actual agent reply from history
    brand: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenExample":
        return cls(**d)


def build_golden_set(
    conversations: List[dict],
    taxonomy: List[Intent],
    llm_client,
    brand: str,
    max_examples: int = 200,
    out_path: Optional[str] = None,
) -> List[GoldenExample]:
    """Build golden evaluation examples from historical conversations.

    Args:
        conversations: list of conversation dicts.
        taxonomy: the derived intent taxonomy.
        llm_client: an LLMClient instance.
        brand: brand to filter conversations for.
        max_examples: maximum number of examples to generate.
        out_path: optional path to save the golden set JSON.

    Returns:
        List of GoldenExample instances.
    """
    # Filter to target brand
    brand_convs = [c for c in conversations if c["brand"] == brand]
    logger.info("Building golden set from %d %s conversations", len(brand_convs), brand)

    examples: List[GoldenExample] = []
    for conv in brand_convs[:max_examples]:
        messages = conv["messages"]

        # Customer's first message is the test input
        customer_msgs = [m for m in messages if m["role"] == "customer"]
        agent_msgs = [m for m in messages if m["role"] == "agent"]

        if not customer_msgs or not agent_msgs:
            continue

        customer_message = customer_msgs[0]["text"]
        reference_reply = agent_msgs[0]["text"]

        # Classify the customer message to get the "expected" intent
        intent_result = classify_intent(
            message=customer_message,
            taxonomy=taxonomy,
            llm_client=llm_client,
            brand=brand,
        )

        example = GoldenExample(
            conversation_id=conv["conversation_id"],
            customer_message=customer_message,
            expected_intent=intent_result.intent_name,
            intent_confidence=intent_result.confidence,
            reference_reply=reference_reply,
            brand=brand,
        )
        examples.append(example)
        logger.debug(
            "Golden example: %s → intent=%s",
            customer_message[:50], intent_result.intent_name,
        )

    logger.info("Built %d golden examples", len(examples))

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([e.to_dict() for e in examples], f, indent=2, default=str)
        logger.info("Saved golden set to %s", out)

    return examples


def load_golden_set(path: str) -> List[GoldenExample]:
    """Load a previously saved golden set from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [GoldenExample.from_dict(d) for d in data]
