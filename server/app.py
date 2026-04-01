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

import json
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
from models import Action, Observation, EpisodeState, ResetResponse, StepResponse, StateResponse

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
    return {"status": "healthy", "service": "sentinelops", "version": "1.0.0"}


@app.get("/", tags=["infra"])
async def root():
    """Root endpoint — redirects to docs in a browser."""
    return {
        "service": "SentinelOps OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/metadata", tags=["openenv"])
async def metadata():
    """
    Return environment metadata (name, description, version).

    Required by the OpenEnv standard for environment discovery.
    """
    return {
        "name": "sentinelops",
        "description": (
            "SentinelOps is a multi-step agentic reinforcement-learning environment "
            "that simulates an AI security analyst monitoring surveillance camera feeds. "
            "The agent must inspect frames, navigate temporal sequences, switch between "
            "cameras, detect anomalies, classify threat severity, and decide whether "
            "to escalate incidents or dismiss false alarms."
        ),
        "version": "1.0.0",
        "author": "Team Adaptrix",
        "license": "MIT",
        "tags": [
            "surveillance", "incident-response", "multi-step",
            "agentic", "reinforcement-learning", "security",
        ],
    }


@app.get("/schema", tags=["openenv"])
async def schema():
    """
    Return JSON schemas for the environment's action, observation, and state.

    Required by OpenEnv for agent integration and validation.
    """
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EpisodeState.model_json_schema(),
    }


@app.post("/mcp", tags=["openenv"])
async def mcp_endpoint(request: Request):
    """
    Minimal MCP (Model Context Protocol) JSON-RPC 2.0 endpoint.

    Supports method discovery and basic ping for OpenEnv validation.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
        )

    rpc_id = body.get("id")
    method = body.get("method", "")

    # Method dispatch
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "sentinelops", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Start or restart an episode",
                    "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}},
                },
                {
                    "name": "step",
                    "description": "Submit an agent action",
                    "inputSchema": Action.model_json_schema(),
                },
                {
                    "name": "state",
                    "description": "Get current episode state",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }
    elif method == "tools/call":
        tool_name = (body.get("params") or {}).get("name", "")
        tool_args = (body.get("params") or {}).get("arguments", {})
        try:
            if tool_name == "reset":
                obs, info = _env.reset(task_id=tool_args.get("task_id"))
                result = {"content": [{"type": "text", "text": json.dumps({"observation": obs.model_dump(), "info": info})}]}
            elif tool_name == "step":
                action = Action(**tool_args)
                obs, reward, terminated, truncated, info = _env.step(action)
                result = {"content": [{"type": "text", "text": json.dumps({"observation": obs.model_dump(), "reward": reward, "terminated": terminated, "truncated": truncated, "info": info})}]}
            elif tool_name == "state":
                state = _env.state()
                result = {"content": [{"type": "text", "text": json.dumps(state.model_dump())}]}
            else:
                return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}, "id": rpc_id})
        except Exception as exc:
            return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}, "id": rpc_id})
    elif method == "ping":
        result = {}
    else:
        return JSONResponse(
            content={"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": rpc_id},
        )

    return JSONResponse(content={"jsonrpc": "2.0", "result": result, "id": rpc_id})


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the server (used by project.scripts entry point)."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
    )


if __name__ == "__main__":
    main()
