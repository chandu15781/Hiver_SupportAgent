"""
Tests for src.data.cleaner.

Run with: pytest tests/test_cleaner.py
(This sandbox has no pytest installed, so these were also verified
manually via scripts/_run_tests_manually.py — see README for the
real `pytest` command to use once dependencies are installed.)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaner import clean_tweets


def _make_df(rows):
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def test_removes_duplicate_tweet_ids():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "hello", "response_tweet_id": None, "in_response_to_tweet_id": None},
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "hello", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    cleaned, report = clean_tweets(df)
    assert len(cleaned) == 1
    assert report.removed_duplicates == 1


def test_removes_empty_text():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "   ", "response_tweet_id": None, "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "real message", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    cleaned, report = clean_tweets(df)
    assert len(cleaned) == 1
    assert report.removed_empty_text == 1
    assert cleaned.iloc[0]["text"] == "real message"


def test_removes_missing_timestamp():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": None,
         "text": "hello", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    cleaned, report = clean_tweets(df)
    assert len(cleaned) == 0
    assert report.removed_missing_timestamp == 1


def test_removes_mention_only_spam():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "@AppleSupport @someoneelse", "response_tweet_id": None, "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "@AppleSupport my phone is broken", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    cleaned, report = clean_tweets(df)
    assert len(cleaned) == 1
    assert report.removed_mention_only == 1
    assert "broken" in cleaned.iloc[0]["text"]


def test_keeps_clean_rows_untouched():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01",
         "text": "a valid customer message", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    cleaned, report = clean_tweets(df)
    assert len(cleaned) == 1
    assert report.ending_rows == 1
    assert report.starting_rows == 1


ALL_TESTS = [
    test_removes_duplicate_tweet_ids,
    test_removes_empty_text,
    test_removes_missing_timestamp,
    test_removes_mention_only_spam,
    test_keeps_clean_rows_untouched,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"PASS: {t.__name__}")
