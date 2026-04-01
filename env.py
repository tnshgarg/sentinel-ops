"""
SentinelOps — Core Environment Engine.

Implements the full multi-step agentic environment with Gymnasium-style
reset / step / state API.  The environment:

  • Loads a task (ground-truth JSON) and its frame sequence.
  • Presents observations (camera frames + context) to the agent.
  • Accepts actions, transitions state, and computes shaped rewards.
  • Terminates on escalation / dismissal / step-limit truncation.

Design patterns used:
  - State Pattern   → episode state machine
  - Strategy Pattern → reward shaping delegated to RewardEngine
  - Factory Pattern  → TaskFactory for loading tasks
"""

from __future__ import annotations

import base64
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DEFAULT_REWARD_TABLE, FRAMES_DIR, SEQUENCES_DIR, TASKS_DIR, settings
from models import (
    Action,
    ActionType,
    AlertLevel,
    EpisodeState,
    FrameAnnotation,
    Observation,
    Reward,
    TaskDifficulty,
    TaskGroundTruth,
)

logger = logging.getLogger("sentinelops.env")


# ---------------------------------------------------------------------------
# Frame Loading Utilities
# ---------------------------------------------------------------------------

# Maps (camera_id, descriptor) -> actual filename on disk.
# In production this would be a database or object store query.
_FRAME_MAP: Dict[str, str] = {
    "cam01_normal": "cam01_normal.png",
    "cam01_anomaly": "cam01_anomaly.png",
    "cam02_suspicious": "cam02_suspicious.png",
    "cam03_corridor": "cam03_corridor.png",
    "cam03_intruder": "cam03_intruder.png",
    "cam04_entrance": "cam04_entrance.png",
}

# Which on-disk frame to use for each (camera_id, anomaly_present) pair.
_CAMERA_FRAME_LOOKUP: Dict[Tuple[str, bool], str] = {
    ("cam-01", False): "cam01_normal.png",
    ("cam-01", True):  "cam01_anomaly.png",
    ("cam-02", False): "cam02_suspicious.png",
    ("cam-02", True):  "cam02_suspicious.png",
    ("cam-03", False): "cam03_corridor.png",
    ("cam-03", True):  "cam03_intruder.png",
    ("cam-04", False): "cam04_entrance.png",
    ("cam-04", True):  "cam04_entrance.png",
}


def _load_frame_b64(task_id: str, frame_id: str, camera_id: str, anomaly_present: bool) -> str:
    """Return base-64 encoded frame bytes for a camera / anomaly combo.
    Checks task-specific sequence frames first, falls back to static frames.
    """
    seq_path = SEQUENCES_DIR / task_id / f"{frame_id}.png"
    if seq_path.exists():
        with open(seq_path, "rb") as fp:
            return base64.b64encode(fp.read()).decode("ascii")
            
    # Fallback logic for static frames
    filename = _CAMERA_FRAME_LOOKUP.get(
        (camera_id, anomaly_present),
        _CAMERA_FRAME_LOOKUP.get((camera_id, False), "cam01_normal.png"),
    )
    path = FRAMES_DIR / filename
    if not path.exists():
        # Fallback: create a tiny placeholder PNG so the env never crashes.
        logger.warning("Frame file %s not found; returning empty placeholder.", path)
        return _generate_placeholder_b64(camera_id)
    with open(path, "rb") as fp:
        return base64.b64encode(fp.read()).decode("ascii")


def _generate_placeholder_b64(label: str = "NO SIGNAL") -> str:
    """Generate a minimal 1×1 transparent PNG as a last-resort placeholder."""
    # Minimal valid PNG (1×1 transparent pixel)
    import struct, zlib
    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat = _png_chunk(b"IDAT", raw)
    iend = _png_chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode("ascii")


# ---------------------------------------------------------------------------
# Task Factory
# ---------------------------------------------------------------------------

class TaskFactory:
    """Loads and caches TaskGroundTruth objects from the tasks/ directory."""

    _cache: Dict[str, TaskGroundTruth] = {}

    @classmethod
    def load_all(cls) -> List[TaskGroundTruth]:
        """Discover and load every task JSON under TASKS_DIR."""
        tasks: List[TaskGroundTruth] = []
        for json_path in sorted(TASKS_DIR.rglob("*.json")):
            task = cls._load_one(json_path)
            if task:
                tasks.append(task)
        return tasks

    @classmethod
    def load_by_id(cls, task_id: str) -> Optional[TaskGroundTruth]:
        """Load a specific task by its ID (searches all difficulty dirs)."""
        if task_id in cls._cache:
            return cls._cache[task_id]
        for json_path in TASKS_DIR.rglob("*.json"):
            task = cls._load_one(json_path)
            if task and task.task_id == task_id:
                return task
        return None

    @classmethod
    def load_by_difficulty(cls, difficulty: TaskDifficulty) -> List[TaskGroundTruth]:
        """Load all tasks at a given difficulty tier."""
        return [t for t in cls.load_all() if t.difficulty == difficulty]

    @classmethod
    def _load_one(cls, path: Path) -> Optional[TaskGroundTruth]:
        if str(path) in cls._cache:
            return cls._cache[str(path)]
        try:
            with open(path) as fp:
                data = json.load(fp)
            task = TaskGroundTruth(**data)
            cls._cache[task.task_id] = task
            cls._cache[str(path)] = task
            return task
        except Exception as exc:
            logger.error("Failed to load task from %s: %s", path, exc)
            return None


# ---------------------------------------------------------------------------
# Reward Engine  (Strategy Pattern)
# ---------------------------------------------------------------------------

class RewardEngine:
    """
    Computes shaped, deterministic rewards based on the agent's action,
    the current episode state, and the task ground-truth.

    Rewards are accumulated across the trajectory (delayed reward).
    """

    def __init__(self, reward_table: Optional[Dict[str, float]] = None):
        self.table = reward_table or DEFAULT_REWARD_TABLE

    def compute(
        self,
        action: Action,
        state: EpisodeState,
        gt: TaskGroundTruth,
        current_frame: FrameAnnotation,
    ) -> Reward:
        score = 0.0
        feedback_parts: List[str] = []
        done = False

        at = action.action_type

        # --- Inspect current frame ---
        if at == ActionType.INSPECT_CURRENT_FRAME:
            if current_frame.anomaly_present:
                score += self.table["correct_anomaly_detection"]
                feedback_parts.append("Anomaly detected correctly on inspection.")
            else:
                # Inspecting a clean frame is neutral (information gathering)
                feedback_parts.append("Frame inspected — no anomaly found.")
            # Penalise redundant inspections of the same frame
            frame_key = f"{state.current_camera}:{state.current_frame_idx}"
            if frame_key in state.frames_inspected:
                score += self.table["redundant_inspect"]
                feedback_parts.append("Redundant re-inspection penalised.")

        # --- Temporal navigation ---
        elif at in (ActionType.REQUEST_PREVIOUS_FRAME, ActionType.REQUEST_NEXT_FRAME):
            # Correct temporal reasoning: moving to the anomaly start frame
            target_idx = (
                state.current_frame_idx - 1
                if at == ActionType.REQUEST_PREVIOUS_FRAME
                else state.current_frame_idx + 1
            )
            if 0 <= target_idx < gt.total_frames:
                target_frame = gt.frames[target_idx]
                if target_frame.anomaly_present:
                    score += self.table["correct_temporal_reasoning"]
                    feedback_parts.append("Good temporal reasoning — anomaly frame reached.")
                else:
                    feedback_parts.append("Navigated to a non-anomaly frame.")
            else:
                feedback_parts.append("Boundary reached — no frame in that direction.")

        # --- Switch camera ---
        elif at == ActionType.SWITCH_CAMERA:
            target_cam = action.payload or ""
            if target_cam == gt.correct_camera and target_cam != state.current_camera:
                score += self.table["correct_camera_switch"]
                feedback_parts.append(f"Switched to correct camera {target_cam}.")
            elif target_cam == state.current_camera:
                score += self.table["random_action_spam"]
                feedback_parts.append("Already on that camera — redundant switch.")
            else:
                feedback_parts.append(f"Switched to camera {target_cam}.")

        # --- Zoom region ---
        elif at == ActionType.ZOOM_REGION:
            region = action.payload or ""
            if current_frame.anomaly_present and region == current_frame.anomaly_region:
                score += self.table["correct_anomaly_detection"] * 0.5
                feedback_parts.append(f"Zoomed into anomaly region '{region}'.")
            else:
                score += self.table["unnecessary_zoom"]
                feedback_parts.append("Zoom did not reveal additional anomaly.")

        # --- Classify risk ---
        elif at == ActionType.CLASSIFY_RISK:
            classified = action.payload or ""
            if classified == gt.correct_risk_level.value:
                score += self.table["correct_risk_classification"]
                feedback_parts.append(f"Risk correctly classified as '{classified}'.")
            else:
                score += self.table["false_positive"] * 0.5
                feedback_parts.append(
                    f"Risk classification '{classified}' incorrect "
                    f"(expected '{gt.correct_risk_level.value}')."
                )

        # --- Escalate ---
        elif at == ActionType.ESCALATE_INCIDENT:
            if gt.should_escalate:
                score += self.table["correct_escalation"]
                feedback_parts.append("Incident correctly escalated.")
            else:
                score += self.table["false_positive"]
                feedback_parts.append("False escalation — this was not a real threat.")
            done = True

        # --- Dismiss ---
        elif at == ActionType.DISMISS_ALERT:
            if not gt.should_escalate:
                score += self.table["correct_escalation"]
                feedback_parts.append("Alert correctly dismissed — no threat found.")
            else:
                score += self.table["unsafe_dismissal"]
                feedback_parts.append("UNSAFE DISMISSAL — a real threat was missed!")
                score += self.table["missed_anomaly"]
            done = True

        # --- Speed bonus ---
        if done and state.current_step + 1 <= gt.optimal_steps:
            score += self.table["fast_response"]
            feedback_parts.append("Speed bonus awarded — resolved within optimal steps.")

        # --- Anti-spam: penalise if recent actions repeat identically ---
        if len(state.action_history) >= 3:
            recent = [h["action_type"] for h in state.action_history[-3:]]
            if len(set(recent)) == 1 and recent[0] == at:
                score += self.table["random_action_spam"]
                feedback_parts.append("Repetitive action pattern detected.")

        cumulative = state.cumulative_reward + score

        return Reward(
            score=round(score, 4),
            feedback=" | ".join(feedback_parts) if feedback_parts else "No specific feedback.",
            done=done,
            cumulative_score=round(cumulative, 4),
        )


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class SentinelOpsEnvironment:
    """
    The SentinelOps multi-step agentic environment.

    Lifecycle:
        1. reset(task_id)  → initial Observation
        2. step(action)    → (Observation, reward, terminated, truncated, info)
        3. state()         → current EpisodeState  (for debugging / logging)
    """

    def __init__(self) -> None:
        self._gt: Optional[TaskGroundTruth] = None
        self._state: Optional[EpisodeState] = None
        self._reward_engine = RewardEngine()
        self._all_tasks = TaskFactory.load_all()
        logger.info("Environment initialised with %d tasks.", len(self._all_tasks))

    # ---- Public API --------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Tuple[Observation, Dict[str, Any]]:
        """
        Start (or restart) an episode.

        Args:
            task_id: Specific task to load.  If ``None``, a random task is chosen.

        Returns:
            (observation, info) — the first observation and episode metadata.
        """
        if task_id:
            gt = TaskFactory.load_by_id(task_id)
            if gt is None:
                raise ValueError(f"Unknown task_id '{task_id}'")
        else:
            if not self._all_tasks:
                self._all_tasks = TaskFactory.load_all()
            gt = random.choice(self._all_tasks)

        self._gt = gt
        initial_camera = gt.camera_ids[0]
        self._state = EpisodeState(
            task_id=gt.task_id,
            current_camera=initial_camera,
            current_frame_idx=0,
            cameras_visited=[initial_camera],
        )

        obs = self._build_observation()
        info = {
            "task_id": gt.task_id,
            "difficulty": gt.difficulty.value,
            "title": gt.title,
            "description": gt.description,
            "total_frames": gt.total_frames,
            "max_steps": gt.max_steps,
            "camera_ids": gt.camera_ids,
        }
        logger.info("Episode reset → task=%s difficulty=%s", gt.task_id, gt.difficulty.value)
        return obs, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._state is None or self._gt is None:
            raise RuntimeError("Environment not initialised — call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode already finished — call reset() to start a new one.")

        gt = self._gt
        state = self._state

        # Current frame before action
        current_frame = gt.frames[min(state.current_frame_idx, len(gt.frames) - 1)]

        # Compute reward
        reward = self._reward_engine.compute(action, state, gt, current_frame)

        # Apply state transitions
        self._apply_transition(action)

        # Record action in history
        state.action_history.append({
            "step": state.current_step,
            "action_type": action.action_type,
            "payload": action.payload,
            "confidence": action.confidence,
            "reward": reward.score,
        })

        state.current_step += 1
        state.cumulative_reward = reward.cumulative_score

        # Determine termination
        terminated = reward.done
        truncated = state.current_step >= gt.max_steps and not terminated
        if terminated or truncated:
            state.done = True
            if truncated:
                # Penalise for not resolving within time limit
                state.cumulative_reward += DEFAULT_REWARD_TABLE["missed_anomaly"] * 0.5

        obs = self._build_observation()
        info = {
            "step": state.current_step,
            "cumulative_reward": state.cumulative_reward,
            "action_taken": action.action_type,
            "feedback": reward.feedback,
        }

        logger.debug(
            "step=%d action=%s reward=%.3f cum=%.3f done=%s",
            state.current_step,
            action.action_type,
            reward.score,
            state.cumulative_reward,
            state.done,
        )

        return obs, reward.score, terminated, truncated, info

    def state(self) -> EpisodeState:
        """Return a copy of the current episode state."""
        if self._state is None:
            raise RuntimeError("No active episode.")
        return self._state.model_copy(deep=True)

    def get_current_task(self) -> Optional[TaskGroundTruth]:
        """Return the ground-truth for the active task (for grading)."""
        return self._gt

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return metadata for every available task."""
        return [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty.value,
                "title": t.title,
                "tags": t.tags,
            }
            for t in self._all_tasks
        ]

    # ---- Internal ----------------------------------------------------------

    def _apply_transition(self, action: Action) -> None:
        """Mutate ``self._state`` according to the action taken."""
        state = self._state
        gt = self._gt
        assert state is not None and gt is not None

        at = action.action_type

        if at == ActionType.INSPECT_CURRENT_FRAME:
            frame_key = f"{state.current_camera}:{state.current_frame_idx}"
            if frame_key not in state.frames_inspected:
                state.frames_inspected.append(frame_key)

        elif at == ActionType.REQUEST_PREVIOUS_FRAME:
            if state.current_frame_idx > 0:
                state.current_frame_idx -= 1
            state.zoom_active = False
            state.zoom_region = None

        elif at == ActionType.REQUEST_NEXT_FRAME:
            if state.current_frame_idx < gt.total_frames - 1:
                state.current_frame_idx += 1
            state.zoom_active = False
            state.zoom_region = None

        elif at == ActionType.SWITCH_CAMERA:
            target = action.payload or state.current_camera
            if target in gt.camera_ids:
                state.current_camera = target
                if target not in state.cameras_visited:
                    state.cameras_visited.append(target)
                # Find the first frame belonging to this camera
                for idx, f in enumerate(gt.frames):
                    if f.camera_id == target:
                        state.current_frame_idx = idx
                        break
            state.zoom_active = False
            state.zoom_region = None

        elif at == ActionType.ZOOM_REGION:
            state.zoom_active = True
            state.zoom_region = action.payload

        elif at == ActionType.CLASSIFY_RISK:
            state.risk_classified = action.payload

        elif at == ActionType.ESCALATE_INCIDENT:
            state.escalated = True

        elif at == ActionType.DISMISS_ALERT:
            state.dismissed = True

    def _build_observation(self) -> Observation:
        """Construct the observation visible to the agent."""
        state = self._state
        gt = self._gt
        assert state is not None and gt is not None

        frame_ann = gt.frames[min(state.current_frame_idx, len(gt.frames) - 1)]
        frame_b64 = _load_frame_b64(
            gt.task_id,
            frame_ann.frame_id,
            frame_ann.camera_id,
            frame_ann.anomaly_present
        )

        # Determine which actions are currently legal
        available = self._legal_actions()

        # Infer alert level from frame
        if frame_ann.anomaly_present:
            alert = AlertLevel.HIGH
        elif state.current_step == 0:
            alert = AlertLevel.MEDIUM  # initial alert triggered the episode
        else:
            alert = AlertLevel.LOW

        context_parts = [frame_ann.description]
        if state.zoom_active and state.zoom_region:
            context_parts.append(f"[ZOOMED into region: {state.zoom_region}]")
        if state.risk_classified:
            context_parts.append(f"[Risk classified as: {state.risk_classified}]")

        return Observation(
            task_id=gt.task_id,
            step=state.current_step,
            camera_id=state.current_camera,
            frame_b64=frame_b64,
            context=" ".join(context_parts),
            available_actions=[a.value for a in available],
            alert_level=alert,
            metadata={
                "frame_id": frame_ann.frame_id,
                "timestamp": frame_ann.timestamp,
                "cameras_visited": state.cameras_visited,
                "zoom": state.zoom_active,
            },
        )

    def _legal_actions(self) -> List[ActionType]:
        """Determine which actions may be taken in the current state."""
        state = self._state
        gt = self._gt
        assert state is not None and gt is not None

        actions: List[ActionType] = [
            ActionType.INSPECT_CURRENT_FRAME,
        ]

        if state.current_frame_idx > 0:
            actions.append(ActionType.REQUEST_PREVIOUS_FRAME)
        if state.current_frame_idx < gt.total_frames - 1:
            actions.append(ActionType.REQUEST_NEXT_FRAME)
        if len(gt.camera_ids) > 1:
            actions.append(ActionType.SWITCH_CAMERA)

        actions.append(ActionType.ZOOM_REGION)
        actions.append(ActionType.CLASSIFY_RISK)
        actions.append(ActionType.ESCALATE_INCIDENT)
        actions.append(ActionType.DISMISS_ALERT)

        return actions
