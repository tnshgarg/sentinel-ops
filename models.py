"""
SentinelOps — Pydantic Models for the Incident Response OpenEnv Environment.

Defines all data contracts: Observation, Action, Reward, Task ground-truth,
and episode metadata structures used by the environment, grader, and inference.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AlertLevel(str, enum.Enum):
    """Severity tiers for incoming surveillance alerts."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, enum.Enum):
    """Agent-assigned risk classification for an observed anomaly."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class ActionType(str, enum.Enum):
    """Exhaustive set of actions the agent may take at each step."""
    INSPECT_CURRENT_FRAME = "inspect_current_frame"
    REQUEST_PREVIOUS_FRAME = "request_previous_frame"
    REQUEST_NEXT_FRAME = "request_next_frame"
    SWITCH_CAMERA = "switch_camera"
    ZOOM_REGION = "zoom_region"
    CLASSIFY_RISK = "classify_risk"
    ESCALATE_INCIDENT = "escalate_incident"
    DISMISS_ALERT = "dismiss_alert"


class TaskDifficulty(str, enum.Enum):
    """Difficulty tiers used for the task ladder."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Core Data Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    What the agent *sees* at each environment step.

    Attributes:
        task_id:            Unique identifier for the current task / episode.
        step:               Current step index within the episode (0-based).
        camera_id:          Identifier of the camera feed currently shown.
        frame_b64:          Base-64 encoded JPEG/PNG of the current frame.
        context:            Optional textual description or alert note.
        available_actions:  Actions the agent may legally choose right now.
        alert_level:        Severity of the most recent alert, if any.
        metadata:           Arbitrary extra key-value metadata.
    """
    task_id: str
    step: int = Field(ge=0)
    camera_id: str
    frame_b64: str
    context: Optional[str] = None
    available_actions: List[str]
    alert_level: Optional[AlertLevel] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """
    An agent's decision submitted to the environment.

    Attributes:
        action_type:  One of the valid ActionType values.
        payload:      Optional extra data (e.g. camera ID for switch, region for zoom).
        confidence:   Agent's self-reported confidence in [0, 1].
    """
    action_type: str
    payload: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("action_type")
    @classmethod
    def _validate_action_type(cls, v: str) -> str:
        valid = {e.value for e in ActionType}
        if v not in valid:
            raise ValueError(f"Invalid action_type '{v}'. Must be one of {sorted(valid)}")
        return v


class Reward(BaseModel):
    """
    Feedback returned to the agent after each step.

    Attributes:
        score:     Immediate numeric reward for the current step.
        feedback:  Human-readable explanation of the reward.
        done:      Whether the episode has terminated.
        cumulative_score:  Running total reward across the episode so far.
    """
    score: float
    feedback: str
    done: bool
    cumulative_score: float = 0.0


# ---------------------------------------------------------------------------
# Task / Ground-Truth Definitions
# ---------------------------------------------------------------------------

class FrameAnnotation(BaseModel):
    """Metadata for a single frame in a task sequence."""
    frame_id: str
    camera_id: str
    description: str
    anomaly_present: bool = False
    anomaly_type: Optional[str] = None
    anomaly_region: Optional[str] = None  # e.g. "top-left", "center"
    timestamp: str = ""


class TaskGroundTruth(BaseModel):
    """
    Complete ground-truth for scoring a single task episode.

    The grader compares the agent's action trajectory to these expected values.
    """
    task_id: str
    difficulty: TaskDifficulty
    title: str
    description: str
    camera_ids: List[str]
    correct_camera: str
    correct_risk_level: RiskLevel
    should_escalate: bool
    anomaly_start_frame: int = Field(ge=0)
    total_frames: int = Field(ge=1)
    frames: List[FrameAnnotation]
    optimal_steps: int = Field(ge=1, description="Minimum steps an ideal agent would need")
    max_steps: int = Field(ge=1, description="Maximum allowed steps before truncation")
    tags: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Episode / State Tracking
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """Mutable runtime state maintained by the environment for one episode."""
    task_id: str
    current_step: int = 0
    current_camera: str = ""
    current_frame_idx: int = 0
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False
    zoom_active: bool = False
    zoom_region: Optional[str] = None
    cameras_visited: List[str] = Field(default_factory=list)
    frames_inspected: List[str] = Field(default_factory=list)
    risk_classified: Optional[str] = None
    escalated: bool = False
    dismissed: bool = False


# ---------------------------------------------------------------------------
# OpenEnv API Response Wrappers
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    """Payload returned by POST /reset."""
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """Payload returned by POST /step."""
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Payload returned by GET /state."""
    episode: EpisodeState
    task_metadata: Dict[str, Any] = Field(default_factory=dict)
