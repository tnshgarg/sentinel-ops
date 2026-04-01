"""
SentinelOps — FastAPI Application Server.

Exposes the OpenEnv-compliant HTTP API:

    POST /reset   → start / restart an episode
    POST /step    → submit an action
    GET  /state   → retrieve current episode state
    GET  /tasks   → list available tasks
    POST /grade   → grade a completed episode
    GET  /health  → liveness probe

Deployed as a Hugging Face Space (Docker) and validated via ``openenv validate``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from env import SentinelOpsEnvironment
from grader import grade
from models import Action, ResetResponse, StepResponse, StateResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-22s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinelops.server")

# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SentinelOps — Incident Response Control Room OpenEnv",
    description=(
        "A multi-step agentic reinforcement-learning environment that "
        "simulates an AI security analyst monitoring surveillance feeds, "
        "detecting anomalies, and escalating incidents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_env = SentinelOpsEnvironment()


# ---------------------------------------------------------------------------
# Request / Response DTOs
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description="Specific task to load.  Omit or null for a random task.",
    )


class StepRequest(BaseModel):
    action_type: str
    payload: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

@app.get("/health", tags=["infra"])
async def health():
    """Liveness probe — always 200 if the server is up."""
    return {"status": "ok", "service": "sentinelops", "version": "1.0.0"}


@app.get("/", tags=["infra"])
async def root():
    """Root endpoint — redirects to docs in a browser."""
    return {
        "service": "SentinelOps OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# OpenEnv Core API
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["openenv"], response_model=None)
async def reset_endpoint(body: Optional[ResetRequest] = None):
    """
    Start or restart an episode.

    Returns the initial observation and task metadata.
    Responds with HTTP 200 as required by OpenEnv validation.
    """
    task_id = body.task_id if body else None
    try:
        observation, info = _env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return JSONResponse(
        status_code=200,
        content={
            "observation": observation.model_dump(),
            "info": info,
        },
    )


@app.post("/step", tags=["openenv"], response_model=None)
async def step_endpoint(body: StepRequest):
    """
    Submit an agent action and advance the environment by one step.

    Returns (observation, reward, terminated, truncated, info).
    """
    try:
        action = Action(
            action_type=body.action_type,
            payload=body.payload,
            confidence=body.confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        obs, reward, terminated, truncated, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return JSONResponse(
        status_code=200,
        content={
            "observation": obs.model_dump(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        },
    )


@app.get("/state", tags=["openenv"], response_model=None)
async def state_endpoint():
    """Return the current episode state (for debugging / transparency)."""
    try:
        episode_state = _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    gt = _env.get_current_task()
    meta = {}
    if gt:
        meta = {
            "task_id": gt.task_id,
            "difficulty": gt.difficulty.value,
            "title": gt.title,
        }

    return JSONResponse(
        status_code=200,
        content={
            "episode": episode_state.model_dump(),
            "task_metadata": meta,
        },
    )


# ---------------------------------------------------------------------------
# Extended API
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["extended"])
async def list_tasks():
    """List all available tasks with their metadata."""
    return _env.list_tasks()


@app.post("/grade", tags=["extended"])
async def grade_endpoint():
    """
    Grade the current (completed) episode using the deterministic grader.

    Must be called after an episode has terminated or been truncated.
    """
    try:
        state = _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is still running. Complete the episode (escalate / dismiss / reach max steps) before grading.",
        )

    gt = _env.get_current_task()
    if gt is None:
        raise HTTPException(status_code=500, detail="No ground-truth loaded.")

    result = grade(gt, state)
    return result


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )
