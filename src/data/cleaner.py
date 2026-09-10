"""
Cleaning for the raw tweet table.

Operates on a copy — the raw CSV in data/raw/ is never modified.
Removes: duplicate tweet_ids, empty/whitespace-only text, rows with
missing timestamps, and very low-signal "spam-like" rows (e.g. a
tweet that is *only* an @mention with no other content).

Each removal reason is logged with a count so the effect of cleaning
is auditable rather than silent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

MENTION_ONLY_RE = re.compile(r"^(@\w+[\s,]*)+$")


@dataclass
class CleaningReport:
    starting_rows: int = 0
    removed_duplicates: int = 0
    removed_empty_text: int = 0
    removed_missing_timestamp: int = 0
    removed_mention_only: int = 0
    ending_rows: int = 0
    details: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "starting_rows": self.starting_rows,
            "removed_duplicates": self.removed_duplicates,
            "removed_empty_text": self.removed_empty_text,
            "removed_missing_timestamp": self.removed_missing_timestamp,
            "removed_mention_only": self.removed_mention_only,
            "ending_rows": self.ending_rows,
        }


def clean_tweets(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(starting_rows=len(df))
    out = df.copy()

    # 1. Duplicate tweet_ids (malformed / re-ingested records)
    before = len(out)
    out = out.drop_duplicates(subset=["tweet_id"], keep="first")
    report.removed_duplicates = before - len(out)

    # 2. Empty or whitespace-only text
    before = len(out)
    normalized_text = out["text"].fillna("").str.strip()
    out = out[normalized_text.str.len() > 0]
    report.removed_empty_text = before - len(out)

    # 3. Missing/unparseable timestamp (malformed record)
    before = len(out)
    out = out[out["created_at"].notna()]
    report.removed_missing_timestamp = before - len(out)

    # 4. Mention-only spam (e.g. "@handle @handle" with nothing else)
    before = len(out)
    is_mention_only = out["text"].fillna("").str.strip().str.match(MENTION_ONLY_RE)
    out = out[~is_mention_only]
    report.removed_mention_only = before - len(out)

    report.ending_rows = len(out)
    return out.reset_index(drop=True), report


if __name__ == "__main__":
    from src.data.loader import load_raw_tweets

    df = load_raw_tweets("data/raw/sample.csv")
    cleaned, report = clean_tweets(df)
    print(report.as_dict())
