"""
Reconstruct conversation threads from individual tweets.

The raw data is a flat table of tweets linked by `in_response_to_tweet_id`
(the tweet this one replies to) and `response_tweet_id` (comma-separated
ids of tweets that reply to this one). Neither field alone gives a full
conversation — we have to walk the reply chain.

Algorithm:
1. Build a lookup: tweet_id -> row.
2. Find "root" customer tweets: inbound=True tweets whose
   in_response_to_tweet_id is null (nothing in the data replies TO them
   from before) — i.e. the customer initiating contact.
3. For each root, walk forward following response_tweet_id links,
   alternating customer/agent turns, until no further replies exist.
   This produces a linear thread. (Branching replies — e.g. a tweet
   with two response_tweet_ids — are resolved by preferring the
   earliest-timestamped continuation, since a real support conversation
   is a single back-and-forth; the alternate branch, if any, is dropped
   for thread purposes but the tweet itself is not deleted from raw data.)
4. Each thread's `brand` is the first non-null brand-handle author_id
   encountered in the thread (a brand author_id is any author_id that
   is NOT purely numeric, since customer author_ids in this dataset
   are pseudonymized numeric strings).
5. Only threads containing at least one customer message AND one
   agent message are considered valid "conversations" for downstream
   use (a lone unanswered tweet is not a conversation).

Output: normalized conversation dicts matching the spec's schema:
    {
      "conversation_id": ...,
      "brand": ...,
      "messages": [{"role": "customer"|"agent", "text": ..., "timestamp": ...}, ...]
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


def _is_brand_handle(author_id: str) -> bool:
    """Customer author_ids in this dataset are numeric strings; brand
    author_ids are handles like 'AppleSupport'. Not foolproof for every
    possible dataset variant, but correct for this schema."""
    return not str(author_id).isdigit()


def _parse_response_ids(raw: object) -> List[int]:
    if pd.isna(raw) or raw == "":
        return []
    return [int(x) for x in str(raw).split(",") if x.strip().isdigit()]


@dataclass
class ThreadBuildReport:
    total_tweets: int = 0
    root_customer_tweets: int = 0
    valid_conversations: int = 0
    dropped_no_agent_reply: int = 0
    dropped_no_brand_identified: int = 0


def build_threads(df: pd.DataFrame) -> tuple[List[dict], ThreadBuildReport]:
    report = ThreadBuildReport(total_tweets=len(df))

    by_id: Dict[int, dict] = {
        int(row.tweet_id): row._asdict() for row in df.itertuples(index=False)
    }

    # Roots: inbound (customer) tweets with no in_response_to_tweet_id,
    # i.e. the customer is opening the conversation.
    roots = [
        row for row in by_id.values()
        if row["inbound"] and pd.isna(row["in_response_to_tweet_id"])
    ]
    report.root_customer_tweets = len(roots)

    conversations: List[dict] = []

    for root in roots:
        thread_rows = [root]
        current = root
        visited = {int(current["tweet_id"])}

        while True:
            next_ids = _parse_response_ids(current["response_tweet_id"])
            next_ids = [tid for tid in next_ids if tid in by_id and tid not in visited]
            if not next_ids:
                break
            # Prefer the earliest-timestamped continuation if multiple exist.
            candidates = [by_id[tid] for tid in next_ids]
            candidates.sort(key=lambda r: r["created_at"] if pd.notna(r["created_at"]) else pd.Timestamp.max.tz_localize("UTC"))
            nxt = candidates[0]
            thread_rows.append(nxt)
            visited.add(int(nxt["tweet_id"]))
            current = nxt

        has_agent_reply = any(not r["inbound"] for r in thread_rows)
        if not has_agent_reply:
            report.dropped_no_agent_reply += 1
            continue

        brand = next(
            (r["author_id"] for r in thread_rows if not r["inbound"] and _is_brand_handle(r["author_id"])),
            None,
        )
        if brand is None:
            report.dropped_no_brand_identified += 1
            continue

        thread_rows.sort(key=lambda r: r["created_at"])
        conversation = {
            "conversation_id": f"conv_{root['tweet_id']}",
            "brand": brand,
            "messages": [
                {
                    "role": "customer" if r["inbound"] else "agent",
                    "text": r["text"],
                    "timestamp": r["created_at"].isoformat() if pd.notna(r["created_at"]) else None,
                    "tweet_id": int(r["tweet_id"]),
                }
                for r in thread_rows
            ],
        }
        conversations.append(conversation)

    report.valid_conversations = len(conversations)
    return conversations, report


if __name__ == "__main__":
    from src.data.loader import load_raw_tweets
    from src.data.cleaner import clean_tweets

    df = load_raw_tweets("data/raw/sample.csv")
    cleaned, _ = clean_tweets(df)
    conversations, report = build_threads(cleaned)

    print(report)
    print(f"\nExample conversation:")
    import json
    print(json.dumps(conversations[0], indent=2, default=str))
