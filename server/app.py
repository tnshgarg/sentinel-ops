"""
SentinelOps — FastAPI Application Server.

Exposes the OpenEnv-compliant HTTP API:

    POST /reset   → start / restart an episode
    POST /step    → submit an action
    GET  /state   → retrieve current episode state
    GET  /tasks   → list available tasks
    POST /grade   → grade a completed episode
    GET  /health  → liveness probe
    GET  /metrics → aggregate performance statistics

Session management:
    Each request may include an optional ``session_id`` field (or query param).
    Omitting it uses the default shared session, preserving full backward
    compatibility with the OpenEnv inference spec and existing tests.

Deployed as a Hugging Face Space (Docker) and validated via ``openenv validate``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
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
    version="1.1.0",
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
# Session State
# ---------------------------------------------------------------------------

DEFAULT_SESSION_ID = "default"

# session_id → SentinelOpsEnvironment
_sessions: Dict[str, SentinelOpsEnvironment] = {}

# Completed grade results for /metrics aggregation
_completed_grades: List[Dict[str, Any]] = []


def _get_session(session_id: str) -> SentinelOpsEnvironment:
    """Return the environment for a given session, creating it if needed."""
    if session_id not in _sessions:
        _sessions[session_id] = SentinelOpsEnvironment()
        logger.info("Created new session: %s", session_id)
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response DTOs
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description="Specific task to load. Omit or null for a random task.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier. Omit to use the default shared session.",
    )


class StepRequest(BaseModel):
    action_type: str
    payload: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier. Omit to use the default shared session.",
    )


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
    return {
        "status": "ok",
        "service": "sentinelops",
        "version": "1.1.0",
        "active_sessions": len(_sessions),
    }


@app.get("/", tags=["infra"])
async def root():
    """Root endpoint — redirects to docs in a browser."""
    return {
        "service": "SentinelOps OpenEnv",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
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
    session_id = (body.session_id if body else None) or DEFAULT_SESSION_ID

    env = _get_session(session_id)
    try:
        observation, info = env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return JSONResponse(
        status_code=200,
        content={
            "observation": observation.model_dump(),
            "info": info,
            "session_id": session_id,
        },
    )


@app.post("/step", tags=["openenv"], response_model=None)
async def step_endpoint(body: StepRequest):
    """
    Submit an agent action and advance the environment by one step.

    Returns (observation, reward, terminated, truncated, info).
    """
    session_id = body.session_id or DEFAULT_SESSION_ID
    env = _get_session(session_id)

    try:
        action = Action(
            action_type=body.action_type,
            payload=body.payload,
            confidence=body.confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        obs, reward, terminated, truncated, info = env.step(action)
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
            "session_id": session_id,
        },
    )


@app.get("/state", tags=["openenv"], response_model=None)
async def state_endpoint(session_id: str = Query(default=DEFAULT_SESSION_ID)):
    """Return the current episode state (for debugging / transparency)."""
    env = _get_session(session_id)
    try:
        episode_state = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    gt = env.get_current_task()
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
            "session_id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# Extended API
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["extended"])
async def list_tasks():
    """List all available tasks with their metadata."""
    # Use default session env to list tasks (task list is global)
    env = _get_session(DEFAULT_SESSION_ID)
    return env.list_tasks()


@app.post("/grade", tags=["extended"])
async def grade_endpoint(session_id: str = Query(default=DEFAULT_SESSION_ID)):
    """
    Grade the current (completed) episode using the deterministic grader.

    Must be called after an episode has terminated or been truncated.
    """
    env = _get_session(session_id)
    try:
        state = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is still running. Complete the episode (escalate / dismiss / reach max steps) before grading.",
        )

    gt = env.get_current_task()
    if gt is None:
        raise HTTPException(status_code=500, detail="No ground-truth loaded.")

    result = grade(gt, state)
    result["session_id"] = session_id

    # Record for /metrics
    _completed_grades.append({
        "task_id": result.get("task_id"),
        "difficulty": result.get("difficulty"),
        "score": result.get("score", 0.0),
        "steps_taken": result.get("steps_taken", 0),
        "optimal_steps": result.get("optimal_steps", 0),
    })

    return result


@app.get("/metrics", tags=["extended"])
async def metrics_endpoint():
    """
    Aggregate performance statistics across all graded episodes.

    Returns per-difficulty breakdowns, average scores, and efficiency metrics.
    Useful for benchmarking agent performance and environment health checks.
    """
    if not _completed_grades:
        return {
            "total_episodes": 0,
            "average_score": 0.0,
            "by_difficulty": {},
            "active_sessions": len(_sessions),
            "message": "No completed episodes yet. Run episodes and call /grade to populate metrics.",
        }

    total = len(_completed_grades)
    total_score = sum(g["score"] for g in _completed_grades)

    # Group by difficulty
    by_difficulty: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_score": 0.0, "total_steps": 0, "total_optimal": 0}
    )
    for g in _completed_grades:
        diff = g.get("difficulty", "unknown")
        by_difficulty[diff]["count"] += 1
        by_difficulty[diff]["total_score"] += g["score"]
        by_difficulty[diff]["total_steps"] += g.get("steps_taken", 0)
        by_difficulty[diff]["total_optimal"] += g.get("optimal_steps", 0)

    diff_summary = {}
    for diff, stats in by_difficulty.items():
        n = stats["count"]
        diff_summary[diff] = {
            "episodes": n,
            "average_score": round(stats["total_score"] / n, 4),
            "average_steps": round(stats["total_steps"] / n, 2),
            "average_optimal_steps": round(stats["total_optimal"] / n, 2),
            "efficiency_ratio": round(
                stats["total_optimal"] / max(stats["total_steps"], 1), 3
            ),
        }

    return {
        "total_episodes": total,
        "average_score": round(total_score / total, 4),
        "best_score": max(g["score"] for g in _completed_grades),
        "worst_score": min(g["score"] for g in _completed_grades),
        "by_difficulty": diff_summary,
        "active_sessions": len(_sessions),
    }


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
