"""
Loader for the raw "Customer Support on Twitter" dataset.

Actual schema (verified against the provided sample, matches the
Kaggle thoughtvector/customer-support-on-twitter dataset):

    tweet_id                 int    unique tweet id
    author_id                str    numeric string for customers (pseudonymized),
                                     brand handle string for companies (e.g. "AppleSupport")
    inbound                   bool   True if authored by a customer, False if by a brand
    created_at                str    tweet timestamp, e.g. "Wed Oct 11 06:55:44 +0000 2017"
    text                      str    tweet body (may contain embedded newlines)
    response_tweet_id         str    comma-separated list of tweet_ids that reply to this tweet
    in_response_to_tweet_id   float  the single tweet_id this tweet is replying to (NaN if none)

We do NOT invent columns beyond these. Any code assuming a different
schema should be treated as a bug.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def load_raw_tweets(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the raw tweet CSV and apply minimal, non-destructive type coercion.

    Raises a ValueError if the file's columns don't match what the rest
    of the pipeline expects, so schema drift fails loudly instead of
    silently producing garbage downstream.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            "tweet_id": "Int64",
            "author_id": "string",
            "response_tweet_id": "string",
            "text": "string",
        },
        nrows=max_rows,
    )

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw dataset at {file_path} is missing expected columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}. Inspect the schema before proceeding."
        )

    # inbound sometimes loads as bool already; normalize defensively.
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")

    return df


if __name__ == "__main__":
    df = load_raw_tweets("data/raw/sample.csv")
    print(f"Loaded {len(df)} rows, {df['inbound'].sum()} inbound (customer) tweets")
    print(df.dtypes)
    print(df.head(3))
