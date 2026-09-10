"""
Classify a customer message into an intent from the derived taxonomy.

Uses the LLM with the taxonomy as context to produce a structured
classification result with intent name, confidence, and reasoning.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

from src.intents.taxonomy import Intent

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """\
You are a customer-support intent classifier for the brand "{brand}".

Given the INTENT TAXONOMY below and a CUSTOMER MESSAGE, classify the \
message into exactly one intent category.

INTENT TAXONOMY:
{taxonomy}

CUSTOMER MESSAGE:
"{message}"

Return ONLY valid JSON with these keys:
- "intent_name": the snake_case name from the taxonomy
- "confidence": a float between 0.0 and 1.0 representing your confidence
- "reasoning": a brief (1-2 sentence) explanation of why this intent was chosen

If the message doesn't clearly fit any intent, use the closest match but \
set confidence accordingly (lower). Do NOT invent new intent names.
"""


@dataclass
class IntentResult:
    intent_name: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntentResult":
        return cls(
            intent_name=d["intent_name"],
            confidence=float(d["confidence"]),
            reasoning=d.get("reasoning", ""),
        )


def classify_intent(
    message: str,
    taxonomy: List[Intent],
    llm_client,
    brand: str = "",
) -> IntentResult:
    """Classify a single customer message against the taxonomy.

    Args:
        message: the raw customer message text.
        taxonomy: list of Intent objects (the derived taxonomy).
        llm_client: an LLMClient instance.
        brand: brand name for prompt context.

    Returns:
        IntentResult with intent_name, confidence, and reasoning.
    """
    taxonomy_text = "\n".join(
        f"- {intent.name}: {intent.description}"
        for intent in taxonomy
    )

    prompt = CLASSIFY_PROMPT.format(
        brand=brand,
        taxonomy=taxonomy_text,
        message=message,
    )

    raw = llm_client.chat(
        [
            {"role": "system", "content": "You are a precise JSON-producing assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse classification JSON: %s\nRaw: %s", e, raw[:300])
        return IntentResult(
            intent_name="unknown",
            confidence=0.0,
            reasoning=f"Failed to parse LLM response: {e}",
        )

    # Validate intent_name exists in taxonomy
    valid_names = {intent.name for intent in taxonomy}
    intent_name = parsed.get("intent_name", "unknown")
    if intent_name not in valid_names:
        logger.warning(
            "LLM returned unknown intent '%s'; valid: %s",
            intent_name, valid_names,
        )
        # Keep the result but flag low confidence
        confidence = min(float(parsed.get("confidence", 0.3)), 0.3)
    else:
        confidence = float(parsed.get("confidence", 0.5))

    return IntentResult(
        intent_name=intent_name,
        confidence=confidence,
        reasoning=parsed.get("reasoning", ""),
    )
