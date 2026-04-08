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

import json
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
from models import (
    Action,
    ActionType,
    AlertLevel,
    EpisodeState,
    Observation,
    Reward,
    TaskDifficulty,
    TaskGroundTruth,
)

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

openapi_tags = [
    {
        "name": "openenv",
        "description": "Standard OpenEnv compliant core API endpoints required for environment integration.",
    },
    {
        "name": "extended",
        "description": "Additional SentinelOps specific endpoints for fetching tasks, grading, and metrics.",
    },
    {
        "name": "infra",
        "description": "Health checks, pings, and dashboard UI endpoints.",
    },
]

app = FastAPI(
    title="SentinelOps — Incident Response Control Room OpenEnv",
    description=(
        "A multi-step agentic reinforcement-learning environment that "
        "simulates an AI security analyst monitoring surveillance feeds, "
        "detecting anomalies, and escalating incidents."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_tags=openapi_tags,
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

# 🧠 GLOBAL VIGILANCE STORE (Phase 16 persistence)
# Maps suspect/threat categories to prior risk levels
GLOBAL_VIGILANCE_DATA: Dict[str, str] = {}

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
    # Elite Predictive Grounding (Phase 15 Council)
    predicted_gaze: Optional[List[int]] = None
    velocity_vector: Optional[List[int]] = None


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
        "status": "healthy",
        "service": "sentinelops",
        "version": "1.1.0",
        "active_sessions": len(_sessions),
    }


@app.get("/", tags=["infra"])
async def root():
    """Root endpoint — serves a beautiful dashboard when accessed in a browser."""
    from fastapi.responses import HTMLResponse

    # Build task summary from environment
    try:
        env = _get_session("_dashboard_")
        task_list = env.list_tasks()
        easy_count = sum(1 for t in task_list if t.get("difficulty") == "easy")
        medium_count = sum(1 for t in task_list if t.get("difficulty") == "medium")
        hard_count = sum(1 for t in task_list if t.get("difficulty") == "hard")
        total_tasks = len(task_list)
    except Exception:
        easy_count = medium_count = hard_count = total_tasks = "?"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SentinelOps — AI Surveillance Control Room</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Inter',sans-serif;background:#0a0e17;color:#e2e8f0;min-height:100vh;overflow-x:hidden}}
  .bg-grid{{position:fixed;top:0;left:0;width:100%;height:100%;background-image:
    linear-gradient(rgba(59,130,246,0.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(59,130,246,0.03) 1px,transparent 1px);
    background-size:40px 40px;z-index:0;pointer-events:none}}
  .container{{max-width:1100px;margin:0 auto;padding:2rem;position:relative;z-index:1}}
  .header{{text-align:center;margin-bottom:3rem;animation:fadeIn 0.8s ease}}
  @keyframes fadeIn{{from{{opacity:0;transform:translateY(-20px)}}to{{opacity:1;transform:translateY(0)}}}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.5}}}}
  .shield{{font-size:4rem;margin-bottom:0.5rem;filter:drop-shadow(0 0 20px rgba(59,130,246,0.4))}}
  h1{{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem}}
  .subtitle{{font-size:1.1rem;color:#94a3b8;font-weight:300;max-width:700px;margin:0 auto;line-height:1.6}}
  .badge-row{{display:flex;justify-content:center;gap:0.5rem;margin-top:1rem;flex-wrap:wrap}}
  .badge{{padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:500;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.05)}}
  .badge.green{{border-color:rgba(52,211,153,0.3);color:#34d399}}
  .badge.blue{{border-color:rgba(96,165,250,0.3);color:#60a5fa}}
  .badge.purple{{border-color:rgba(167,139,250,0.3);color:#a78bfa}}
  .live-dot{{width:8px;height:8px;background:#34d399;border-radius:50%;display:inline-block;animation:pulse 2s ease infinite;margin-right:6px;vertical-align:middle}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1.5rem;margin-bottom:2.5rem}}
  .card{{background:linear-gradient(145deg,rgba(15,23,42,0.8),rgba(15,23,42,0.4));border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:1.5rem;backdrop-filter:blur(12px);transition:all 0.3s ease}}
  .card:hover{{border-color:rgba(96,165,250,0.3);transform:translateY(-2px);box-shadow:0 8px 32px rgba(59,130,246,0.1)}}
  .card-icon{{font-size:1.8rem;margin-bottom:0.75rem}}
  .card h3{{font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;color:#f1f5f9}}
  .card p{{font-size:0.875rem;color:#94a3b8;line-height:1.5}}
  .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2.5rem}}
  .stat{{text-align:center;padding:1.25rem;background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px}}
  .stat-number{{font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace}}
  .stat-number.blue{{color:#60a5fa}}
  .stat-number.green{{color:#34d399}}
  .stat-number.amber{{color:#fbbf24}}
  .stat-number.red{{color:#f87171}}
  .stat-label{{font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-top:0.25rem}}
  .actions-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.75rem;margin-top:1rem}}
  .action-chip{{padding:8px 14px;background:rgba(30,41,59,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#94a3b8;transition:all 0.2s}}
  .action-chip:hover{{border-color:rgba(96,165,250,0.4);color:#60a5fa}}
  .api-section{{margin-top:2rem}}
  .api-link{{display:inline-flex;align-items:center;gap:0.5rem;padding:10px 24px;background:linear-gradient(135deg,#3b82f6,#6366f1);color:white;text-decoration:none;border-radius:10px;font-weight:500;font-size:0.95rem;transition:all 0.3s}}
  .api-link:hover{{transform:translateY(-1px);box-shadow:0 4px 20px rgba(59,130,246,0.4)}}
  .footer{{text-align:center;margin-top:3rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.06);color:#475569;font-size:0.85rem}}
  .footer a{{color:#60a5fa;text-decoration:none}}
  @media(max-width:768px){{.stats{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:1.8rem}}.cards{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="container">
  <div class="header">
    <div class="shield">🛡️</div>
    <h1>SentinelOps</h1>
    <p class="subtitle">Interactive Incident Response Control Room — a multi-step agentic RL environment where AI agents monitor surveillance feeds, detect anomalies, and escalate incidents in real time.</p>
    <div class="badge-row">
      <span class="badge green"><span class="live-dot"></span>Live</span>
      <span class="badge blue">OpenEnv Compatible</span>
      <span class="badge purple">Multi-Step Agentic</span>
      <span class="badge">Meta × Scaler Hackathon</span>
    </div>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-number blue">{total_tasks}</div><div class="stat-label">Total Tasks</div></div>
    <div class="stat"><div class="stat-number green">{easy_count}</div><div class="stat-label">Easy</div></div>
    <div class="stat"><div class="stat-number amber">{medium_count}</div><div class="stat-label">Medium</div></div>
    <div class="stat"><div class="stat-number red">{hard_count}</div><div class="stat-label">Hard</div></div>
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-icon">📡</div>
      <h3>Multi-Camera Surveillance</h3>
      <p>Agents navigate between 1–4 camera feeds, tracking suspects across locations. Each camera provides different viewpoints of the same incident timeline.</p>
    </div>
    <div class="card">
      <div class="card-icon">🧠</div>
      <h3>Sequential Decision Making</h3>
      <p>5–15 step episodes with shaped trajectory rewards. Agents must inspect frames, navigate timelines, classify threats, and make escalation decisions.</p>
    </div>
    <div class="card">
      <div class="card-icon">⚖️</div>
      <h3>Deterministic Grading</h3>
      <p>Rubric-based scoring — no LLM-as-judge. Three difficulty-specific graders with reproducible scores between 0.0 and 1.0.</p>
    </div>
    <div class="card">
      <div class="card-icon">👁️</div>
      <h3>Vision-Language Support</h3>
      <p>Sends real CCTV frame images (base64-encoded) to VLM agents. Temporal variation and HUD overlays simulate authentic surveillance footage.</p>
    </div>
  </div>

  <h3 style="margin-bottom:0.75rem;color:#f1f5f9">🎮 Action Space</h3>
  <div class="actions-grid">
    <div class="action-chip">inspect_current_frame</div>
    <div class="action-chip">request_previous_frame</div>
    <div class="action-chip">request_next_frame</div>
    <div class="action-chip">switch_camera</div>
    <div class="action-chip">zoom_region</div>
    <div class="action-chip">classify_risk</div>
    <div class="action-chip">escalate_incident</div>
    <div class="action-chip">dismiss_alert</div>
  </div>

  <div class="api-section" style="text-align:center;margin-top:2rem">
    <a href="/docs" class="api-link">📖 Interactive API Docs</a>
    &nbsp;&nbsp;
    <a href="/redoc" class="api-link" style="background:linear-gradient(135deg,#6366f1,#8b5cf6)">📋 ReDoc</a>
  </div>

  <div class="footer">
    <p>Built by <strong>Team Adaptrix</strong> for the Meta × Scaler OpenEnv Hackathon</p>
    <p style="margin-top:0.4rem">
      <a href="/health">Health</a> · <a href="/tasks">Tasks</a> · <a href="/metrics">Metrics</a> · <a href="/metadata">Metadata</a> · <a href="/schema">Schema</a>
    </p>
  </div>
</div>
</body>
</html>"""

    return HTMLResponse(content=html, status_code=200)


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
        "version": "1.1.0",
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
            mcp_env = _get_session(DEFAULT_SESSION_ID)
            if tool_name == "reset":
                obs, info = mcp_env.reset(task_id=tool_args.get("task_id"))
                result = {"content": [{"type": "text", "text": json.dumps({"observation": obs.model_dump(), "info": info})}]}
            elif tool_name == "step":
                action = Action(**tool_args)
                obs, reward, terminated, truncated, info = mcp_env.step(action)
                result = {"content": [{"type": "text", "text": json.dumps({"observation": obs.model_dump(), "reward": reward, "terminated": terminated, "truncated": truncated, "info": info})}]}
            elif tool_name == "state":
                state = mcp_env.state()
                result = {"content": [{"type": "text", "text": json.dumps(state.model_dump(), default=str)}]}
            else:
                return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}, "id": rpc_id})
            return JSONResponse(content={"jsonrpc": "2.0", "result": result, "id": rpc_id})
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
    session_id = (body.session_id if body else None) or DEFAULT_SESSION_ID

    env = _get_session(session_id)
    try:
        observation, info = env.reset(task_id=task_id)
        
        # 🧠 INJECT GLOBAL VIGILANCE (Prior History Memory)
        # If we have prior history on this facility/suspect type, escalate the posture
        prior_risk = GLOBAL_VIGILANCE_DATA.get(task_id, "safe")
        if prior_risk != "safe":
            info["prior_history"] = f"PRIOR HISTORY: This target was previously classified as '{prior_risk}'."
            if prior_risk in ["dangerous", "critical"]:
                observation.alert_level = AlertLevel.HIGH
                info["tactical_note"] = "VIGILANCE MODE: Persistent threat detected. Safety hardlocks are active."
                
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
            predicted_gaze=body.predicted_gaze,
            velocity_vector=body.velocity_vector
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 🧠 RECORD VIGILANCE (Memory for future episodes)
        if action.action_type == "classify_risk" and action.payload:
            GLOBAL_VIGILANCE_DATA[obs.task_id] = action.payload
            logger.info("Vigilance Recorded: %s -> %s", obs.task_id, action.payload)
            
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
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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
