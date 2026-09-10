"""
Generate a draft reply grounded in retrieved historical conversations (RAG).

The LLM is prompted with:
1. The customer's message and classified intent.
2. The retrieved similar conversations showing how the brand actually
   handled similar issues historically.
3. Instructions to reply in the brand's voice and tone, referencing
   the historical patterns rather than inventing solutions.
"""
from __future__ import annotations

import logging
from typing import List

from src.intents.classifier import IntentResult
from src.retrieval.retriever import RetrievedConversation

logger = logging.getLogger(__name__)

REPLY_PROMPT = """\
You are a customer-support agent for **{brand}** on Twitter. \
Draft a reply to the customer message below.

**IMPORTANT GROUNDING RULES:**
- Base your reply on HOW THIS BRAND actually handled similar issues, \
  as shown in the SIMILAR CONVERSATIONS below.
- Match the brand's tone, phrasing style, and typical response patterns.
- Do NOT invent solutions, URLs, or phone numbers that aren't in the \
  historical examples.
- Keep the reply concise (Twitter-length: under 280 characters if possible, \
  but can be longer if needed for clarity).
- Be empathetic and professional.
- If the historical examples show the brand directing customers to DM, \
  follow that pattern.

CUSTOMER MESSAGE:
"{message}"

CLASSIFIED INTENT: {intent} (confidence: {confidence:.0%})

SIMILAR CONVERSATIONS (how {brand} handled similar issues):
{similar_conversations}

Draft your reply below. Return ONLY the reply text, no JSON, no quotes, \
no commentary.
"""


def _format_similar_conversations(
    retrieved: List[RetrievedConversation],
) -> str:
    """Format retrieved conversations as context for the reply prompt."""
    if not retrieved:
        return "(No similar historical conversations found.)"

    parts = []
    for i, rc in enumerate(retrieved, 1):
        msgs = []
        for msg in rc.messages:
            role_label = "Customer" if msg["role"] == "customer" else "Agent"
            msgs.append(f"  {role_label}: {msg['text']}")
        conv_text = "\n".join(msgs)
        parts.append(
            f"--- Conversation {i} (similarity: {rc.similarity:.2f}) ---\n"
            f"{conv_text}"
        )
    return "\n\n".join(parts)


def generate_reply(
    message: str,
    intent_result: IntentResult,
    retrieved_conversations: List[RetrievedConversation],
    llm_client,
    brand: str = "",
) -> str:
    """Generate a RAG-grounded draft reply to a customer message.

    Args:
        message: the raw customer message.
        intent_result: classification result.
        retrieved_conversations: similar historical conversations for context.
        llm_client: an LLMClient instance.
        brand: the brand name.

    Returns:
        The draft reply text.
    """
    similar_text = _format_similar_conversations(retrieved_conversations)

    prompt = REPLY_PROMPT.format(
        brand=brand,
        message=message,
        intent=intent_result.intent_name,
        confidence=intent_result.confidence,
        similar_conversations=similar_text,
    )

    reply = llm_client.chat(
        [
            {"role": "system",
             "content": f"You are a helpful customer-support agent for {brand}. "
                        f"Reply concisely and professionally."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=512,
    )

    reply = reply.strip().strip('"').strip("'")
    logger.info("Generated reply (%d chars): %s…", len(reply), reply[:100])
    return reply
