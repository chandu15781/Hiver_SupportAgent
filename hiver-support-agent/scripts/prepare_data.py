"""
Phase 1 pipeline: raw tweets -> cleaned tweets -> reconstructed conversations.

Usage:
    python scripts/prepare_data.py --raw-path data/raw/sample.csv --max-rows 50000

Outputs:
    data/processed/cleaned_tweets.csv
    data/processed/conversations.json
    data/processed/cleaning_report.json
    data/processed/thread_build_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_tweets
from src.data.cleaner import clean_tweets
from src.data.thread_builder import build_threads


def main():
    parser = argparse.ArgumentParser(description="Prepare raw tweet data into conversation threads.")
    parser.add_argument("--raw-path", default="data/raw/sample.csv")
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Cap rows read from raw CSV (avoids requiring the full 3M-row dataset).")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw tweets from {args.raw_path} (max_rows={args.max_rows})...")
    df = load_raw_tweets(args.raw_path, max_rows=args.max_rows)
    print(f"  -> {len(df)} rows loaded")

    print("Cleaning...")
    cleaned, clean_report = clean_tweets(df)
    print(f"  -> {clean_report.as_dict()}")

    print("Building conversation threads...")
    conversations, thread_report = build_threads(cleaned)
    print(f"  -> {thread_report}")

    cleaned.to_csv(out_dir / "cleaned_tweets.csv", index=False)
    with open(out_dir / "conversations.json", "w") as f:
        json.dump(conversations, f, indent=2, default=str)
    with open(out_dir / "cleaning_report.json", "w") as f:
        json.dump(clean_report.as_dict(), f, indent=2)
    with open(out_dir / "thread_build_report.json", "w") as f:
        json.dump(thread_report.__dict__, f, indent=2)

    print(f"\nSaved {len(cleaned)} cleaned tweets and {len(conversations)} conversations to {out_dir}/")


if __name__ == "__main__":
    main()
