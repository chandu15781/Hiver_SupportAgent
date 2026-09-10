"""
Analyze reconstructed conversations and recommend a brand for the
project, rather than picking one arbitrarily.

Scores each brand on:
  - conversation_count       (raw volume)
  - customer_message_count
  - agent_message_count
  - resolved_count           (heuristic: thread has >=2 agent turns,
                               a common signal of a worked-through issue,
                               OR ends on an agent message that isn't a
                               generic "please DM us" deflection)
  - avg_thread_length        (proxy for issue complexity / diversity)
  - unique_customers         (breadth, avoids one noisy customer dominating)

These are combined into a simple composite score. This is a heuristic,
not a claim of statistical rigor — it is meant to make brand selection
inspectable and reproducible rather than arbitrary, per the assignment
requirement. The final choice is still a human (project author) decision,
documented in DECISION_LOG.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

# Generic deflection responses that don't represent a "resolution" —
# used only for a soft heuristic on resolution rate.
_DEFLECTION_MARKERS = ("dm us", "send us a dm", "please dm", "dm your", "send a dm")


def _looks_like_pure_deflection(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in _DEFLECTION_MARKERS) and len(text) < 220


@dataclass
class BrandStats:
    brand: str
    conversation_count: int
    customer_message_count: int
    agent_message_count: int
    resolved_count: int
    avg_thread_length: float
    unique_customers: int
    composite_score: float = 0.0


def analyze_brands(conversations: List[dict]) -> pd.DataFrame:
    rows = []
    for conv in conversations:
        messages = conv["messages"]
        customer_msgs = [m for m in messages if m["role"] == "customer"]
        agent_msgs = [m for m in messages if m["role"] == "agent"]

        # Resolution heuristic: at least one agent message that is NOT a
        # pure deflection, OR the thread has multiple agent turns
        # (suggesting real back-and-forth rather than a single canned reply).
        non_deflection_agent_msgs = [
            m for m in agent_msgs if not _looks_like_pure_deflection(m["text"])
        ]
        resolved = len(non_deflection_agent_msgs) > 0 or len(agent_msgs) >= 2

        rows.append({
            "brand": conv["brand"],
            "conversation_id": conv["conversation_id"],
            "customer_msg_count": len(customer_msgs),
            "agent_msg_count": len(agent_msgs),
            "thread_length": len(messages),
            "resolved": resolved,
            "customer_ids": tuple(
                m.get("tweet_id") for m in customer_msgs
            ),  # placeholder; real customer id joined below
        })

    conv_df = pd.DataFrame(rows)
    if conv_df.empty:
        return pd.DataFrame(columns=[
            "brand", "conversation_count", "customer_message_count",
            "agent_message_count", "resolved_count", "avg_thread_length",
            "unique_customers", "composite_score",
        ])

    grouped = conv_df.groupby("brand").agg(
        conversation_count=("conversation_id", "count"),
        customer_message_count=("customer_msg_count", "sum"),
        agent_message_count=("agent_msg_count", "sum"),
        resolved_count=("resolved", "sum"),
        avg_thread_length=("thread_length", "mean"),
    ).reset_index()

    # unique_customers requires the original per-message author, computed separately
    unique_customers = _unique_customers_per_brand(conversations)
    grouped["unique_customers"] = grouped["brand"].map(unique_customers).fillna(0).astype(int)

    # Composite score: normalize each metric to [0,1] and average.
    # Volume matters most (conversation_count, unique_customers), then
    # depth (avg_thread_length) and apparent resolution rate.
    def _norm(col: pd.Series) -> pd.Series:
        rng = col.max() - col.min()
        return (col - col.min()) / rng if rng > 0 else pd.Series([1.0] * len(col), index=col.index)

    grouped["resolution_rate"] = grouped["resolved_count"] / grouped["conversation_count"]
    grouped["composite_score"] = (
        0.35 * _norm(grouped["conversation_count"])
        + 0.20 * _norm(grouped["unique_customers"])
        + 0.15 * _norm(grouped["avg_thread_length"])
        + 0.30 * grouped["resolution_rate"]
    ).round(4)

    return grouped.sort_values("composite_score", ascending=False).reset_index(drop=True)


def _unique_customers_per_brand(conversations: List[dict]) -> dict:
    # A conversation's customer identity isn't stored directly on the
    # conversation dict (only tweet-level ids are), so approximate
    # "unique customers" as unique conversation_ids per brand — each
    # root tweet is one customer-initiated thread. This is a stated
    # simplification, not a true unique-author count.
    counts: dict = {}
    for conv in conversations:
        counts.setdefault(conv["brand"], set()).add(conv["conversation_id"])
    return {brand: len(ids) for brand, ids in counts.items()}


if __name__ == "__main__":
    from src.data.loader import load_raw_tweets
    from src.data.cleaner import clean_tweets
    from src.data.thread_builder import build_threads

    df = load_raw_tweets("data/raw/sample.csv")
    cleaned, _ = clean_tweets(df)
    conversations, _ = build_threads(cleaned)

    ranking = analyze_brands(conversations)
    pd.set_option("display.width", 120)
    print(ranking.to_string(index=False))
