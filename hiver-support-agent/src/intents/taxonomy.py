"""
Derive an intent taxonomy from real conversation data using an LLM.

Instead of imposing a canned list of intents, we show the LLM a
representative sample of customer messages from the selected brand's
historical data and ask it to cluster them into 8–15 intent categories.

This produces a taxonomy that reflects the actual distribution of
customer issues, not a generic one-size-fits-all list.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

TAXONOMY_PROMPT = """\
You are an expert customer-support analyst. Below are real customer \
messages sent to the brand "{brand}" on Twitter.

Analyze these messages and derive a taxonomy of **8–15 intent categories** \
that cover the issues customers are raising. Each category should be:
- Mutually exclusive (a message fits into exactly one)
- Collectively exhaustive (every message below fits into at least one)
- Named with a short, descriptive slug (e.g. "device_performance", \
"billing_issue", "account_access")

For each intent, provide:
- `name`: short snake_case slug
- `description`: 1–2 sentence description of what this intent covers
- `example_messages`: 2–3 verbatim messages from the list below that \
  exemplify this intent

Return ONLY valid JSON — an array of objects with keys: name, description, \
example_messages. No markdown fences, no commentary.

--- CUSTOMER MESSAGES ---
{messages}
"""


@dataclass
class Intent:
    name: str
    description: str
    example_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        return cls(
            name=d["name"],
            description=d["description"],
            example_messages=d.get("example_messages", []),
        )


def build_taxonomy(
    conversations: List[dict],
    llm_client,
    brand: str,
    out_path: Optional[str] = None,
) -> List[Intent]:
    """Derive an intent taxonomy from customer messages in conversations.

    Args:
        conversations: list of conversation dicts (from thread builder).
        llm_client: an LLMClient instance.
        brand: the brand name to filter conversations for.
        out_path: optional path to save the taxonomy JSON.

    Returns:
        A list of Intent dataclass instances.
    """
    # Collect customer messages for the target brand
    customer_messages = []
    for conv in conversations:
        if conv["brand"] != brand:
            continue
        for msg in conv["messages"]:
            if msg["role"] == "customer":
                customer_messages.append(msg["text"])

    if not customer_messages:
        raise ValueError(
            f"No customer messages found for brand '{brand}'. "
            f"Check that conversations.json has data for this brand."
        )

    logger.info(
        "Building intent taxonomy from %d customer messages for %s",
        len(customer_messages), brand,
    )

    # Build the prompt with all customer messages (sample is small enough)
    numbered = "\n".join(
        f"{i+1}. {msg}" for i, msg in enumerate(customer_messages)
    )
    prompt = TAXONOMY_PROMPT.format(brand=brand, messages=numbered)

    raw = llm_client.chat(
        [
            {"role": "system", "content": "You are a precise JSON-producing assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    # Parse the response — handle both bare array and {intents: [...]} wrapper
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse taxonomy JSON: %s\nRaw: %s", e, raw[:500])
        raise ValueError(f"LLM returned invalid JSON for taxonomy: {e}") from e

    if isinstance(parsed, dict):
        # Find the array inside the dict
        for key in ("intents", "taxonomy", "categories", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # If it's a dict without a known key, try the first list value
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON array of intents, got {type(parsed).__name__}"
        )

    intents = [Intent.from_dict(item) for item in parsed]
    logger.info("Derived %d intent categories: %s",
                len(intents), [i.name for i in intents])

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([i.to_dict() for i in intents], f, indent=2)
        logger.info("Saved taxonomy to %s", out)

    return intents


def load_taxonomy(path: str) -> List[Intent]:
    """Load a previously saved taxonomy from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [Intent.from_dict(d) for d in data]
