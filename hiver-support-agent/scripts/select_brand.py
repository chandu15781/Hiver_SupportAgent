"""
Phase 1 pipeline: analyze reconstructed conversations and recommend
a brand to build the agent around.

Usage:
    python scripts/select_brand.py --conversations data/processed/conversations.json

Prints a ranked table and saves it to data/processed/brand_ranking.csv.
Does NOT auto-write the chosen brand into configs/config.yaml — brand
selection is a human decision, documented in DECISION_LOG.md, and the
config should be updated deliberately after reviewing this output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.brand_selector import analyze_brands


def main():
    parser = argparse.ArgumentParser(description="Recommend a brand based on conversation data.")
    parser.add_argument("--conversations", default="data/processed/conversations.json")
    parser.add_argument("--out", default="data/processed/brand_ranking.csv")
    args = parser.parse_args()

    with open(args.conversations) as f:
        conversations = json.load(f)

    ranking = analyze_brands(conversations)
    pd.set_option("display.width", 120)
    print(ranking.to_string(index=False))

    ranking.to_csv(args.out, index=False)
    print(f"\nSaved ranking to {args.out}")

    if len(ranking) > 0:
        top = ranking.iloc[0]
        print(f"\nRecommended brand: {top['brand']} "
              f"(composite_score={top['composite_score']}, "
              f"{int(top['conversation_count'])} conversations)")


if __name__ == "__main__":
    main()
