"""
FastAPI server for the Hiver Support Agent.

Endpoints:
    POST /support     — process a customer message through the full pipeline
    GET  /health      — health check
    GET  /taxonomy    — return the current intent taxonomy

Start with:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Global agent reference — populated at startup
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts and construct the agent at startup."""
    global _agent
    from src.agent.support_agent import SupportAgent
    from src.config import load_config

    config = load_config()
    processed_dir = str(PROJECT_ROOT / "data" / "processed")

    try:
        _agent = SupportAgent.from_artifacts(
            processed_dir=processed_dir,
            config=config,
        )
        logger.info("Support agent loaded successfully")
    except FileNotFoundError as e:
        logger.error(
            "Agent artifacts not found: %s. "
            "Run 'python scripts/build_index.py' first.",
            e,
        )
        _agent = None

    yield

    _agent = None
    logger.info("Server shut down")


app = FastAPI(
    title="Hiver Support Agent",
    description="AI customer-support agent with intent classification, "
                "RAG-grounded replies, and escalation logic.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request / Response models ---

class SupportRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="The customer message to process.",
        json_schema_extra={"example": "My iPhone is running slow after the iOS update"},
    )


class IntentResponse(BaseModel):
    intent_name: str
    confidence: float
    reasoning: str


class EscalationResponse(BaseModel):
    decision: str
    reason: str
    triggered_rules: list[str]


class RetrievedConversationResponse(BaseModel):
    conversation_id: str
    brand: str
    similarity: float


class SupportResponse(BaseModel):
    message: str
    intent: IntentResponse
    escalation: EscalationResponse
    draft_reply: str | None
    retrieved_conversations: list[RetrievedConversationResponse]


class HealthResponse(BaseModel):
    status: str
    agent_loaded: bool


class TaxonomyIntentResponse(BaseModel):
    name: str
    description: str
    example_messages: list[str]


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _agent else "degraded",
        agent_loaded=_agent is not None,
    )


@app.post("/support", response_model=SupportResponse)
async def support(request: SupportRequest):
    """Process a customer message through the full agent pipeline."""
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not loaded. Run 'python scripts/build_index.py' "
                   "to build required artifacts first.",
        )

    try:
        response = _agent.handle(request.message)
    except Exception as e:
        logger.exception("Error processing message")
        raise HTTPException(status_code=500, detail=str(e))

    return SupportResponse(
        message=response.message,
        intent=IntentResponse(**response.intent.to_dict()),
        escalation=EscalationResponse(**response.escalation.to_dict()),
        draft_reply=response.draft_reply,
        retrieved_conversations=[
            RetrievedConversationResponse(
                conversation_id=rc.conversation_id,
                brand=rc.brand,
                similarity=rc.similarity,
            )
            for rc in response.retrieved_conversations
        ],
    )


@app.get("/taxonomy", response_model=list[TaxonomyIntentResponse])
async def taxonomy():
    """Return the current intent taxonomy."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded.")

    return [
        TaxonomyIntentResponse(
            name=intent.name,
            description=intent.description,
            example_messages=intent.example_messages,
        )
        for intent in _agent.taxonomy
    ]


# Mount static directory and UI root
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the interactive web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Hiver Support Agent API is running. Visit /docs for OpenAPI specs."}

