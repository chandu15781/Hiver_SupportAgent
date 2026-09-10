"""
Run the evaluation pipeline:
  1. Load or build the golden test set.
  2. Construct the support agent from saved artifacts.
  3. Run the agent on each golden example.
  4. Compute and display metrics.
  5. Save the evaluation report.

Usage:
    python scripts/run_evaluation.py [--brand AppleSupport]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.llm.client import LLMClient
from src.intents.taxonomy import load_taxonomy
from src.evaluation.golden_set import build_golden_set, load_golden_set
from src.evaluation.evaluator import evaluate_agent
from src.agent.support_agent import SupportAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Evaluate the support agent.")
    parser.add_argument(
        "--brand", default=config.brand.name or "AppleSupport",
    )
    parser.add_argument(
        "--conversations", default="data/processed/conversations.json",
    )
    parser.add_argument(
        "--golden-set", default="data/golden/golden_set.json",
        help="Path to golden set (built if not found).",
    )
    parser.add_argument(
        "--processed-dir", default="data/processed",
    )
    parser.add_argument(
        "--report-dir", default="reports",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge reply scoring (faster, cheaper).",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden_set)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or build golden set
    print(f"\n{'='*60}")
    print(f"STEP 1: Golden test set")
    print(f"{'='*60}")

    llm_client = LLMClient()

    if golden_path.exists():
        print(f"Loading existing golden set from {golden_path}")
        golden_set = load_golden_set(str(golden_path))
    else:
        print(f"Building golden set (not found at {golden_path})...")
        with open(args.conversations) as f:
            conversations = json.load(f)
        taxonomy = load_taxonomy(str(Path(args.processed_dir) / "intent_taxonomy.json"))
        golden_set = build_golden_set(
            conversations=conversations,
            taxonomy=taxonomy,
            llm_client=llm_client,
            brand=args.brand,
            max_examples=config.evaluation.golden_set_size,
            out_path=str(golden_path),
        )

    print(f"Golden set: {len(golden_set)} examples")

    # Step 2: Construct agent
    print(f"\n{'='*60}")
    print(f"STEP 2: Loading support agent")
    print(f"{'='*60}")

    agent = SupportAgent.from_artifacts(
        processed_dir=args.processed_dir,
        config=config,
    )
    print(f"Agent loaded: {agent.brand}, {len(agent.taxonomy)} intents")

    # Step 3: Run evaluation
    print(f"\n{'='*60}")
    print(f"STEP 3: Running evaluation ({len(golden_set)} examples)")
    print(f"{'='*60}")

    report = evaluate_agent(
        agent=agent,
        golden_set=golden_set,
        judge_replies=not args.no_judge,
        out_path=str(report_dir / "evaluation_report.json"),
    )

    # Step 4: Display results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total examples:           {report.total_examples}")
    print(f"Intent accuracy:          {report.intent_accuracy:.2%}")
    print(f"Avg retrieval similarity: {report.avg_retrieval_similarity:.4f}")
    print(f"Avg reply quality:        {report.avg_reply_quality:.2f}/5.0")
    print(f"Auto-handled:             {report.auto_handled_count}")
    print(f"Escalated:                {report.escalated_count}")
    if report.escalation_reasons:
        print(f"Escalation reasons:")
        for reason, count in sorted(
            report.escalation_reasons.items(), key=lambda x: -x[1]
        ):
            print(f"  - {reason}: {count}")

    print(f"\nReport saved to: {report_dir / 'evaluation_report.json'}")
    print(f"LLM usage: {llm_client.usage.log_summary()}")


if __name__ == "__main__":
    main()
