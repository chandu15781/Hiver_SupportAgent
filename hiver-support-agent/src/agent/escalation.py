"""
Escalation decision logic.

Combines deterministic rules with model-informed signals to decide
whether a customer message should be auto-handled or escalated to a
human agent. The decision is always accompanied by an explicit reason
string for auditability.

Escalation triggers (any one is sufficient):
1. Intent confidence is below the configured threshold.
2. Best retrieval similarity is below the configured threshold
   (i.e., we haven't seen a similar issue before).
3. Intent falls into a "always-escalate" category (safety, legal,
   account compromise, etc.).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

from src.intents.classifier import IntentResult
from src.retrieval.retriever import RetrievedConversation

logger = logging.getLogger(__name__)

# Intents that should ALWAYS be escalated regardless of confidence/similarity.
# These represent situations where an automated reply carries real risk.
ALWAYS_ESCALATE_INTENTS = frozenset({
    "account_security",
    "account_compromise",
    "legal_issue",
    "safety_concern",
    "fraud",
    "data_privacy",
    "harassment",
    "physical_harm",
})


@dataclass
class EscalationDecision:
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason: str  # human-readable explanation
    triggered_rules: List[str]  # which rules fired

    def to_dict(self) -> dict:
        return asdict(self)


def decide_escalation(
    intent_result: IntentResult,
    retrieved_conversations: List[RetrievedConversation],
    intent_threshold: float = 0.70,
    retrieval_threshold: float = 0.70,
) -> EscalationDecision:
    """Decide whether to auto-handle or escalate.

    This is deliberately deterministic + transparent: each rule is checked
    independently, and the triggered rules are logged. The decision errs
    on the side of escalation (if ANY rule fires, escalate).

    Args:
        intent_result: classification result from the intent classifier.
        retrieved_conversations: list of retrieved similar conversations.
        intent_threshold: minimum confidence to auto-handle.
        retrieval_threshold: minimum similarity to auto-handle.

    Returns:
        EscalationDecision with decision, reason, and triggered_rules.
    """
    triggered: List[str] = []
    reasons: List[str] = []

    # Rule 1: Always-escalate intent
    if intent_result.intent_name in ALWAYS_ESCALATE_INTENTS:
        triggered.append("always_escalate_intent")
        reasons.append(
            f"Intent '{intent_result.intent_name}' requires human review "
            f"(safety/security/legal category)"
        )

    # Rule 2: Low intent confidence
    if intent_result.confidence < intent_threshold:
        triggered.append("low_intent_confidence")
        reasons.append(
            f"Intent confidence {intent_result.confidence:.2f} is below "
            f"threshold {intent_threshold:.2f} — the system is uncertain "
            f"about the customer's intent"
        )

    # Rule 3: Low retrieval similarity (no good historical precedent)
    best_similarity = 0.0
    if retrieved_conversations:
        best_similarity = max(rc.similarity for rc in retrieved_conversations)

    if best_similarity < retrieval_threshold:
        triggered.append("low_retrieval_similarity")
        reasons.append(
            f"Best retrieval similarity {best_similarity:.2f} is below "
            f"threshold {retrieval_threshold:.2f} — no sufficiently similar "
            f"historical conversation found to ground a reply"
        )

    # Rule 4: No retrieved conversations at all
    if not retrieved_conversations:
        if "low_retrieval_similarity" not in triggered:
            triggered.append("no_retrieved_conversations")
            reasons.append(
                "No historical conversations were retrieved — cannot "
                "ground a reply in past brand behavior"
            )

    # Decision
    if triggered:
        decision = "ESCALATE"
        reason = "Escalated: " + "; ".join(reasons)
    else:
        decision = "AUTO_HANDLE"
        reason = (
            f"Auto-handling: intent '{intent_result.intent_name}' "
            f"(confidence {intent_result.confidence:.2f}), "
            f"best retrieval similarity {best_similarity:.2f}"
        )

    result = EscalationDecision(
        decision=decision,
        reason=reason,
        triggered_rules=triggered,
    )
    logger.info("Escalation decision: %s (%s)", result.decision, result.triggered_rules)
    return result
