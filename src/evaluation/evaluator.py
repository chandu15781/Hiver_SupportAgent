"""
Evaluate the support agent against a golden test set.

Metrics computed:
- **Intent accuracy**: fraction of examples where the agent's classified
  intent matches the golden set's expected intent.
- **Average retrieval similarity**: mean of the best retrieval similarity
  across all examples (how well does the retriever find relevant context).
- **Reply quality (LLM-as-judge)**: an LLM scores each generated reply
  on a 1–5 scale for relevance, helpfulness, and brand-voice consistency.
- **Escalation stats**: counts of auto-handled vs escalated, plus
  analysis of escalation reasons.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from src.agent.support_agent import SupportAgent, AgentResponse
from src.evaluation.golden_set import GoldenExample

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are evaluating a customer-support agent's draft reply.

CUSTOMER MESSAGE: "{customer_message}"
REFERENCE REPLY (actual historical reply): "{reference_reply}"
AGENT'S DRAFT REPLY: "{draft_reply}"

Rate the agent's draft reply on a scale of 1-5 for each criterion:
1. **Relevance**: Does the reply address the customer's actual issue?
2. **Helpfulness**: Does the reply provide actionable information or next steps?
3. **Tone**: Is the reply professional, empathetic, and appropriate?

Return ONLY valid JSON with keys: relevance (int 1-5), helpfulness (int 1-5), \
tone (int 1-5), overall (float average of the three), brief_comment (string).
"""


@dataclass
class ExampleResult:
    """Result of evaluating a single example."""
    conversation_id: str
    customer_message: str
    expected_intent: str
    predicted_intent: str
    intent_correct: bool
    best_retrieval_similarity: float
    escalation_decision: str
    escalation_reason: str
    draft_reply: Optional[str]
    reply_quality: Optional[dict] = None  # LLM-as-judge scores

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Aggregate evaluation metrics."""
    total_examples: int = 0
    intent_accuracy: float = 0.0
    avg_retrieval_similarity: float = 0.0
    avg_reply_quality: float = 0.0
    auto_handled_count: int = 0
    escalated_count: int = 0
    escalation_reasons: dict = field(default_factory=dict)
    example_results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_agent(
    agent: SupportAgent,
    golden_set: List[GoldenExample],
    judge_replies: bool = True,
    out_path: Optional[str] = None,
) -> EvaluationReport:
    """Run the agent on each golden example and compute metrics.

    Args:
        agent: a constructed SupportAgent instance.
        golden_set: list of GoldenExample test cases.
        judge_replies: if True, use LLM-as-judge for reply quality.
        out_path: optional path to save the report JSON.

    Returns:
        EvaluationReport with aggregate metrics and per-example results.
    """
    report = EvaluationReport(total_examples=len(golden_set))
    intent_correct_count = 0
    similarities: List[float] = []
    reply_qualities: List[float] = []
    escalation_reason_counts: dict = {}

    for i, example in enumerate(golden_set):
        logger.info(
            "Evaluating example %d/%d: %s",
            i + 1, len(golden_set), example.customer_message[:50],
        )

        # Run the agent
        response: AgentResponse = agent.handle(example.customer_message)

        # Intent accuracy
        intent_correct = response.intent.intent_name == example.expected_intent
        if intent_correct:
            intent_correct_count += 1

        # Retrieval similarity
        best_sim = 0.0
        if response.retrieved_conversations:
            best_sim = max(rc.similarity for rc in response.retrieved_conversations)
        similarities.append(best_sim)

        # Escalation tracking
        if response.escalation.decision == "AUTO_HANDLE":
            report.auto_handled_count += 1
        else:
            report.escalated_count += 1
            for rule in response.escalation.triggered_rules:
                escalation_reason_counts[rule] = (
                    escalation_reason_counts.get(rule, 0) + 1
                )

        # Reply quality (LLM-as-judge)
        reply_quality = None
        if judge_replies and response.draft_reply:
            reply_quality = _judge_reply(
                customer_message=example.customer_message,
                reference_reply=example.reference_reply,
                draft_reply=response.draft_reply,
                llm_client=agent.llm,
            )
            if reply_quality and "overall" in reply_quality:
                reply_qualities.append(reply_quality["overall"])

        result = ExampleResult(
            conversation_id=example.conversation_id,
            customer_message=example.customer_message,
            expected_intent=example.expected_intent,
            predicted_intent=response.intent.intent_name,
            intent_correct=intent_correct,
            best_retrieval_similarity=best_sim,
            escalation_decision=response.escalation.decision,
            escalation_reason=response.escalation.reason,
            draft_reply=response.draft_reply,
            reply_quality=reply_quality,
        )
        report.example_results.append(result.to_dict())

    # Aggregate metrics
    report.intent_accuracy = (
        intent_correct_count / len(golden_set) if golden_set else 0.0
    )
    report.avg_retrieval_similarity = (
        sum(similarities) / len(similarities) if similarities else 0.0
    )
    report.avg_reply_quality = (
        sum(reply_qualities) / len(reply_qualities) if reply_qualities else 0.0
    )
    report.escalation_reasons = escalation_reason_counts

    logger.info(
        "Evaluation complete: intent_acc=%.2f, avg_sim=%.2f, avg_quality=%.2f, "
        "auto=%d, escalated=%d",
        report.intent_accuracy, report.avg_retrieval_similarity,
        report.avg_reply_quality, report.auto_handled_count, report.escalated_count,
    )

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Saved evaluation report to %s", out)

    return report


def _judge_reply(
    customer_message: str,
    reference_reply: str,
    draft_reply: str,
    llm_client,
) -> Optional[dict]:
    """Use LLM-as-judge to score a draft reply."""
    prompt = JUDGE_PROMPT.format(
        customer_message=customer_message,
        reference_reply=reference_reply,
        draft_reply=draft_reply,
    )

    try:
        raw = llm_client.chat(
            [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        return json.loads(raw)
    except Exception as e:
        logger.warning("Failed to judge reply: %s", e)
        return None
