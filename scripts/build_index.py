"""
Build all artifacts required by the support agent:
  1. Filter conversations for the selected brand.
  2. Embed conversations and build a FAISS index.
  3. Derive an intent taxonomy from customer messages.

Usage:
    python scripts/build_index.py [--brand AppleSupport] [--conversations data/processed/conversations.json]

Outputs:
    data/processed/faiss_index.bin
    data/processed/embeddings.npy
    data/processed/embedding_metadata.json
    data/processed/intent_taxonomy.json
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
from src.retrieval.embedder import embed_conversations
from src.retrieval.index import build_index, save_index
from src.intents.taxonomy import build_taxonomy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Build FAISS index and intent taxonomy.")
    parser.add_argument(
        "--brand", default=config.brand.name or "AppleSupport",
        help="Brand to build artifacts for.",
    )
    parser.add_argument(
        "--conversations", default="data/processed/conversations.json",
        help="Path to conversations JSON.",
    )
    parser.add_argument(
        "--out-dir", default="data/processed",
        help="Output directory for artifacts.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load conversations
    with open(args.conversations) as f:
        all_conversations = json.load(f)

    # Filter to target brand
    brand_conversations = [
        c for c in all_conversations if c["brand"] == args.brand
    ]
    logger.info(
        "Found %d conversations for %s (out of %d total)",
        len(brand_conversations), args.brand, len(all_conversations),
    )

    if not brand_conversations:
        logger.error("No conversations found for brand '%s'. Aborting.", args.brand)
        sys.exit(1)

    # Initialize LLM client
    llm_client = LLMClient()

    # Step 1: Embed conversations and build FAISS index
    print(f"\n{'='*60}")
    print(f"STEP 1: Embedding {len(brand_conversations)} conversations...")
    print(f"{'='*60}")

    embeddings, metadata = embed_conversations(
        brand_conversations, llm_client, out_dir=str(out_dir),
    )

    index = build_index(embeddings)
    save_index(index, str(out_dir / "faiss_index.bin"))

    # Step 2: Build intent taxonomy
    print(f"\n{'='*60}")
    print(f"STEP 2: Deriving intent taxonomy for {args.brand}...")
    print(f"{'='*60}")

    taxonomy = build_taxonomy(
        conversations=all_conversations,
        llm_client=llm_client,
        brand=args.brand,
        out_path=str(out_dir / "intent_taxonomy.json"),
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Brand: {args.brand}")
    print(f"Conversations indexed: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Intent categories: {len(taxonomy)}")
    for intent in taxonomy:
        print(f"  - {intent.name}: {intent.description}")
    print(f"\nArtifacts saved to: {out_dir}/")
    print(f"  - faiss_index.bin")
    print(f"  - embeddings.npy")
    print(f"  - embedding_metadata.json")
    print(f"  - intent_taxonomy.json")
    print(f"\nLLM usage: {llm_client.usage.log_summary()}")
    print(f"\nNext: run the server with 'python -m uvicorn src.api:app --reload'")


if __name__ == "__main__":
    main()
