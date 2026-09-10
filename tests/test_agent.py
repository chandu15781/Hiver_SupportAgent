"""
Tests for the agent module — escalation logic and agent pipeline.

Escalation tests are fully deterministic (no LLM). Pipeline tests use
a mock LLM client.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intents.classifier import IntentResult
from src.intents.taxonomy import Intent
from src.retrieval.retriever import RetrievedConversation
from src.agent.escalation import (
    EscalationDecision,
    decide_escalation,
    ALWAYS_ESCALATE_INTENTS,
)


# ---------------------------------------------------------------
# Escalation tests (deterministic, no LLM)
# ---------------------------------------------------------------

def test_auto_handle_when_all_thresholds_met():
    """High confidence + high similarity → AUTO_HANDLE."""
    intent = IntentResult(
        intent_name="device_performance",
        confidence=0.90,
        reasoning="test",
    )
    retrieved = [
        RetrievedConversation(
            conversation_id="conv_1",
            brand="AppleSupport",
            similarity=0.85,
            query_text="test",
            messages=[],
        )
    ]
    result = decide_escalation(intent, retrieved, intent_threshold=0.70, retrieval_threshold=0.70)
    assert result.decision == "AUTO_HANDLE", f"Expected AUTO_HANDLE, got {result.decision}"
    assert len(result.triggered_rules) == 0


def test_escalate_low_intent_confidence():
    """Low intent confidence → ESCALATE."""
    intent = IntentResult(
        intent_name="device_performance",
        confidence=0.40,
        reasoning="test",
    )
    retrieved = [
        RetrievedConversation(
            conversation_id="conv_1",
            brand="AppleSupport",
            similarity=0.85,
            query_text="test",
            messages=[],
        )
    ]
    result = decide_escalation(intent, retrieved, intent_threshold=0.70, retrieval_threshold=0.70)
    assert result.decision == "ESCALATE"
    assert "low_intent_confidence" in result.triggered_rules


def test_escalate_low_retrieval_similarity():
    """Low retrieval similarity → ESCALATE."""
    intent = IntentResult(
        intent_name="device_performance",
        confidence=0.90,
        reasoning="test",
    )
    retrieved = [
        RetrievedConversation(
            conversation_id="conv_1",
            brand="AppleSupport",
            similarity=0.30,
            query_text="test",
            messages=[],
        )
    ]
    result = decide_escalation(intent, retrieved, intent_threshold=0.70, retrieval_threshold=0.70)
    assert result.decision == "ESCALATE"
    assert "low_retrieval_similarity" in result.triggered_rules


def test_escalate_no_retrieved_conversations():
    """No retrieved conversations → ESCALATE."""
    intent = IntentResult(
        intent_name="device_performance",
        confidence=0.90,
        reasoning="test",
    )
    result = decide_escalation(intent, [], intent_threshold=0.70, retrieval_threshold=0.70)
    assert result.decision == "ESCALATE"
    assert any(
        r in result.triggered_rules
        for r in ("no_retrieved_conversations", "low_retrieval_similarity")
    )


def test_escalate_always_escalate_intent():
    """Always-escalate intent → ESCALATE regardless of confidence."""
    for intent_name in ["account_security", "legal_issue", "fraud"]:
        intent = IntentResult(
            intent_name=intent_name,
            confidence=0.99,
            reasoning="test",
        )
        retrieved = [
            RetrievedConversation(
                conversation_id="conv_1",
                brand="AppleSupport",
                similarity=0.95,
                query_text="test",
                messages=[],
            )
        ]
        result = decide_escalation(
            intent, retrieved, intent_threshold=0.70, retrieval_threshold=0.70
        )
        assert result.decision == "ESCALATE", (
            f"Expected ESCALATE for {intent_name}, got {result.decision}"
        )
        assert "always_escalate_intent" in result.triggered_rules


def test_escalation_multiple_rules():
    """Multiple escalation rules can fire simultaneously."""
    intent = IntentResult(
        intent_name="account_security",
        confidence=0.30,
        reasoning="test",
    )
    result = decide_escalation(intent, [], intent_threshold=0.70, retrieval_threshold=0.70)
    assert result.decision == "ESCALATE"
    assert len(result.triggered_rules) >= 2  # at least always_escalate + low confidence


def test_escalation_decision_has_reason():
    """All decisions must have a non-empty reason string."""
    intent = IntentResult(
        intent_name="device_performance",
        confidence=0.90,
        reasoning="test",
    )
    retrieved = [
        RetrievedConversation(
            conversation_id="conv_1",
            brand="AppleSupport",
            similarity=0.85,
            query_text="test",
            messages=[],
        )
    ]
    auto = decide_escalation(intent, retrieved, intent_threshold=0.70, retrieval_threshold=0.70)
    assert auto.reason, "AUTO_HANDLE should have a reason"

    intent_low = IntentResult(intent_name="x", confidence=0.1, reasoning="test")
    esc = decide_escalation(intent_low, [], intent_threshold=0.70, retrieval_threshold=0.70)
    assert esc.reason, "ESCALATE should have a reason"


# ---------------------------------------------------------------
# Intent result / taxonomy dataclass tests
# ---------------------------------------------------------------

def test_intent_result_to_dict():
    ir = IntentResult(intent_name="billing", confidence=0.8, reasoning="matches billing keywords")
    d = ir.to_dict()
    assert d["intent_name"] == "billing"
    assert d["confidence"] == 0.8


def test_intent_from_dict():
    intent = Intent.from_dict({
        "name": "billing",
        "description": "Billing issues",
        "example_messages": ["charge me twice"],
    })
    assert intent.name == "billing"
    assert len(intent.example_messages) == 1


# ---------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------

def _run_tests():
    tests = [
        test_auto_handle_when_all_thresholds_met,
        test_escalate_low_intent_confidence,
        test_escalate_low_retrieval_similarity,
        test_escalate_no_retrieved_conversations,
        test_escalate_always_escalate_intent,
        test_escalation_multiple_rules,
        test_escalation_decision_has_reason,
        test_intent_result_to_dict,
        test_intent_from_dict,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)
