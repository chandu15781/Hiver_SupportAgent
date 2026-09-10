"""
Tests for src.data.thread_builder.

Run with: pytest tests/test_thread_builder.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.thread_builder import build_threads


def _make_df(rows):
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def test_builds_simple_two_turn_conversation():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01T10:00:00Z",
         "text": "my order is late", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "BrandCo", "inbound": False, "created_at": "2020-01-01T10:05:00Z",
         "text": "sorry, DM us your order number", "response_tweet_id": None, "in_response_to_tweet_id": 1},
    ])
    conversations, report = build_threads(df)
    assert report.valid_conversations == 1
    conv = conversations[0]
    assert conv["brand"] == "BrandCo"
    assert [m["role"] for m in conv["messages"]] == ["customer", "agent"]


def test_drops_root_with_no_agent_reply():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01T10:00:00Z",
         "text": "anyone there?", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    conversations, report = build_threads(df)
    assert report.valid_conversations == 0
    assert report.dropped_no_agent_reply == 1


def test_follows_multi_turn_thread():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01T10:00:00Z",
         "text": "my order is late", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "BrandCo", "inbound": False, "created_at": "2020-01-01T10:05:00Z",
         "text": "what is your order number?", "response_tweet_id": "3", "in_response_to_tweet_id": 1},
        {"tweet_id": 3, "author_id": "1001", "inbound": True, "created_at": "2020-01-01T10:10:00Z",
         "text": "it is 55512", "response_tweet_id": "4", "in_response_to_tweet_id": 2},
        {"tweet_id": 4, "author_id": "BrandCo", "inbound": False, "created_at": "2020-01-01T10:15:00Z",
         "text": "thanks, refunding now", "response_tweet_id": None, "in_response_to_tweet_id": 3},
    ])
    conversations, report = build_threads(df)
    assert report.valid_conversations == 1
    conv = conversations[0]
    assert len(conv["messages"]) == 4
    assert [m["role"] for m in conv["messages"]] == ["customer", "agent", "customer", "agent"]


def test_picks_earliest_branch_on_forked_reply():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01T10:00:00Z",
         "text": "issue here", "response_tweet_id": "2,3", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "BrandCo", "inbound": False, "created_at": "2020-01-01T10:20:00Z",
         "text": "later reply", "response_tweet_id": None, "in_response_to_tweet_id": 1},
        {"tweet_id": 3, "author_id": "BrandCo", "inbound": False, "created_at": "2020-01-01T10:05:00Z",
         "text": "earlier reply", "response_tweet_id": None, "in_response_to_tweet_id": 1},
    ])
    conversations, report = build_threads(df)
    assert report.valid_conversations == 1
    conv = conversations[0]
    assert conv["messages"][1]["text"] == "earlier reply"


def test_brand_is_identified_from_non_numeric_author_id():
    df = _make_df([
        {"tweet_id": 1, "author_id": "1001", "inbound": True, "created_at": "2020-01-01T10:00:00Z",
         "text": "help", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "SomeBrandHandle", "inbound": False, "created_at": "2020-01-01T10:05:00Z",
         "text": "how can we help?", "response_tweet_id": None, "in_response_to_tweet_id": 1},
    ])
    conversations, report = build_threads(df)
    assert conversations[0]["brand"] == "SomeBrandHandle"


ALL_TESTS = [
    test_builds_simple_two_turn_conversation,
    test_drops_root_with_no_agent_reply,
    test_follows_multi_turn_thread,
    test_picks_earliest_branch_on_forked_reply,
    test_brand_is_identified_from_non_numeric_author_id,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"PASS: {t.__name__}")
