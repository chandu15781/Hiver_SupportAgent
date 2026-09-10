"""
Central configuration for the Hiver Support Agent.

Loads settings from configs/config.yaml and allows override via
environment variables (see .env.example). Nothing in this project
should hardcode a brand name, file path, or model name outside of
this module's defaults.

NOTE: implemented with stdlib dataclasses rather than pydantic.
See DECISION_LOG.md: the target deployment stack uses pydantic
(as specified), but this dev sandbox has no network access to
install it, so dataclasses are used here to keep the pipeline
runnable and testable end-to-end. Swapping to pydantic.BaseModel
is a drop-in change (same field names) when running in an
environment with the real dependencies installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DatasetConfig:
    raw_path: str = "data/raw/sample.csv"
    max_rows: Optional[int] = None
    sample_size: int = 50_000


@dataclass
class BrandConfig:
    name: Optional[str] = None


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.75


@dataclass
class ClassificationConfig:
    confidence_threshold: float = 0.70


@dataclass
class EscalationConfig:
    retrieval_threshold: float = 0.70
    intent_threshold: float = 0.70


@dataclass
class EvaluationConfig:
    golden_set_size: int = 200
    human_agreement_sample_size: int = 40


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key_env_var: str = "OPENAI_API_KEY"


@dataclass
class EmbeddingConfig:
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))


@dataclass
class ProjectConfig:
    name: str = "hiver-support-agent"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str] = None) -> ProjectConfig:
    """Load config from YAML, falling back to defaults for any missing keys."""
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        return ProjectConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return ProjectConfig(
        name=raw.get("project", {}).get("name", "hiver-support-agent"),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        brand=BrandConfig(**raw.get("brand", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        classification=ClassificationConfig(**raw.get("classification", {})),
        escalation=EscalationConfig(**raw.get("escalation", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg.to_dict(), indent=2))
