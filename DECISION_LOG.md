# Decision Log

Non-obvious decisions made during implementation, and why.

## Phase 1 — Data

1. **Config implemented with stdlib `dataclasses` instead of `pydantic` for now.**
   Reason: the dev sandbox used to build/test this project has no network
   access, so `pydantic` (and `fastapi`/`faiss`) can't be installed here.
   `requirements.txt` still pins pydantic as the target dependency; the
   dataclass version uses identical field names so swapping is mechanical
   once run in an environment with the real dependencies.

2. **Thread reconstruction walks `response_tweet_id` forward from customer
   root tweets rather than walking `in_response_to_tweet_id` backward.**
   Reason: roots (conversation starts) are unambiguous — a customer tweet
   with no `in_response_to_tweet_id` — while walking backward from every
   tweet would require re-deriving the same roots anyway. Forward-walking
   from roots also makes branching explicit and handled in one place.

3. **On a forked reply (`response_tweet_id` lists more than one continuation),
   the earliest-timestamped branch is kept and the others are dropped from
   the thread (but not deleted from the raw/cleaned data).**
   Reason: a real support conversation is a linear back-and-forth; multiple
   replies to the same tweet in this dataset are usually either a duplicate
   agent response or an unrelated third party jumping into the thread.
   This is a heuristic and is flagged as a limitation for the "what's
   misleading" section later.

4. **A brand is identified per-thread as the first non-numeric `author_id`
   among agent messages**, since customer `author_id`s in this dataset are
   pseudonymized numeric strings and brand `author_id`s are handles
   (e.g. `AppleSupport`). This is specific to this dataset's schema.

5. **Brand selection score = weighted blend of conversation volume (35%),
   unique customers/threads (20%), avg thread depth (15%), and a resolution-rate
   heuristic (30%)**, rather than raw conversation count alone.
   Reason: assignment requires justifying brand choice on diversity/quality,
   not just volume; a resolution heuristic (non-deflection agent replies,
   or 2+ agent turns) is a cheap proxy for "conversations that actually
   got worked through" without needing manual labeling at this stage.
   This heuristic is acknowledged as approximate, not ground truth.

6. **Working dataset for this phase is the provided `sample.csv` (~100 rows),
   not the full 3M-row Kaggle dataset.** The loader/cleaner/thread-builder/
   brand-selector are all written against the dataset's real schema
   (verified from the sample) and parameterized by `--max-rows` /
   `dataset.sample_size` in config, so they scale to a larger file without
   code changes. Brand-ranking results on this tiny sample (AppleSupport
   recommended, 11 conversations) should NOT be taken as representative —
   this will be re-run once/if a larger sample is available, and this
   caveat will be carried into the report's "what's misleading" section.

7. **Cleaning removes duplicate `tweet_id`s, empty/whitespace-only text,
   rows with unparseable timestamps, and "mention-only" spam** (a tweet
   that is nothing but @handles). Kept deliberately conservative — nothing
   is dropped based on message length, language, or sentiment, since
   those are legitimate variation, not data-quality problems.
