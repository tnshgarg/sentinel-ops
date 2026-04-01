"""
SentinelOps — Deterministic Grader Module.

Implements strategy-pattern graders for evaluating agent trajectories
against task ground-truth.  NO LLM-as-judge — purely rubric-based scoring.

Grader hierarchy:
    BaseGrader (ABC)
      ├── EasyGrader     —  single-camera anomaly detection
      ├── MediumGrader   —  multi-frame temporal reasoning
      └── HardGrader     —  multi-camera coordinated incident

The top-level ``grade()`` function auto-selects the appropriate strategy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from models import (
    ActionType,
    EpisodeState,
    RiskLevel,
    TaskDifficulty,
    TaskGroundTruth,
)

logger = logging.getLogger("sentinelops.grader")


# ---------------------------------------------------------------------------
# Base Grader (Strategy interface)
# ---------------------------------------------------------------------------

class BaseGrader(ABC):
    """
    Abstract grader that evaluates an agent's episode trajectory.

    Subclasses implement ``evaluate()`` to return a score ∈ [0, 1]
    along with a detailed breakdown dictionary.
    """

    @abstractmethod
    def evaluate(
        self,
        gt: TaskGroundTruth,
        state: EpisodeState,
    ) -> Dict[str, Any]:
        """
        Score the completed episode.

        Returns:
            dict with at-minimum keys ``"score"`` (float 0–1) and
            ``"breakdown"`` (dict of component scores).
        """
        raise NotImplementedError

    # ---- Shared helpers ----------------------------------------------------

    @staticmethod
    def _action_types_in_history(state: EpisodeState) -> List[str]:
        return [h["action_type"] for h in state.action_history]

    @staticmethod
    def _payloads_for_action(state: EpisodeState, action_type: str) -> List[Optional[str]]:
        return [
            h.get("payload")
            for h in state.action_history
            if h["action_type"] == action_type
        ]

    @staticmethod
    def _has_action(state: EpisodeState, action_type: str) -> bool:
        return any(h["action_type"] == action_type for h in state.action_history)


# ---------------------------------------------------------------------------
# Easy Grader
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Scoring for single-camera anomaly detection tasks.

    Rubric (max 1.0):
        ✓ Inspected frame            0.15
        ✓ Detected anomaly frame     0.25
        ✓ Correct risk classification 0.30
        ✓ Correct escalation / dismiss 0.30
    """

    def evaluate(self, gt: TaskGroundTruth, state: EpisodeState) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        # 1. Did the agent inspect any frame?
        if self._has_action(state, ActionType.INSPECT_CURRENT_FRAME):
            breakdown["frame_inspection"] = 0.15
            score += 0.15
        else:
            breakdown["frame_inspection"] = 0.0

        # 2. Did the agent navigate to / stay on a frame with an anomaly?
        anomaly_frames = {f.frame_id for f in gt.frames if f.anomaly_present}
        inspected_anomaly = any(
            h.get("action_type") == ActionType.INSPECT_CURRENT_FRAME
            for h in state.action_history
            # We also check frames_inspected which tracks camera:idx keys
        )
        # Check via frames_inspected set
        anomaly_detected = False
        for key in state.frames_inspected:
            cam, idx_str = key.split(":")
            idx = int(idx_str)
            if idx < len(gt.frames) and gt.frames[idx].anomaly_present:
                anomaly_detected = True
                break

        if anomaly_detected:
            breakdown["anomaly_detection"] = 0.25
            score += 0.25
        else:
            breakdown["anomaly_detection"] = 0.0

        # 3. Correct risk classification
        risk_payloads = self._payloads_for_action(state, ActionType.CLASSIFY_RISK)
        if risk_payloads and risk_payloads[-1] == gt.correct_risk_level.value:
            breakdown["risk_classification"] = 0.30
            score += 0.30
        else:
            breakdown["risk_classification"] = 0.0

        # 4. Correct escalation / dismissal
        if gt.should_escalate and state.escalated:
            breakdown["escalation_decision"] = 0.30
            score += 0.30
        elif not gt.should_escalate and state.dismissed:
            breakdown["escalation_decision"] = 0.30
            score += 0.30
        else:
            breakdown["escalation_decision"] = 0.0

        final = min(score, 1.0)
        return {
            "score": round(final, 4),
            "max_score": 1.0,
            "breakdown": breakdown,
            "grader": "EasyGrader",
        }


# ---------------------------------------------------------------------------
# Medium Grader
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Scoring for multi-frame temporal reasoning tasks.

    Rubric (max 1.0):
        ✓ Inspected frames                 0.10
        ✓ Correct temporal navigation       0.20
        ✓ Anomaly onset identified          0.20
        ✓ Correct risk classification       0.25
        ✓ Correct escalation / dismiss      0.25
    """

    def evaluate(self, gt: TaskGroundTruth, state: EpisodeState) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        # 1. Frame inspection (at least 2 different frames inspected)
        if len(state.frames_inspected) >= 2:
            breakdown["frame_inspection"] = 0.10
            score += 0.10
        elif len(state.frames_inspected) == 1:
            breakdown["frame_inspection"] = 0.05
            score += 0.05
        else:
            breakdown["frame_inspection"] = 0.0

        # 2. Temporal navigation — did the agent use prev/next frame actions?
        nav_actions = [
            ActionType.REQUEST_PREVIOUS_FRAME,
            ActionType.REQUEST_NEXT_FRAME,
        ]
        nav_count = sum(
            1 for h in state.action_history if h["action_type"] in nav_actions
        )
        if nav_count >= 2:
            breakdown["temporal_navigation"] = 0.20
            score += 0.20
        elif nav_count == 1:
            breakdown["temporal_navigation"] = 0.10
            score += 0.10
        else:
            breakdown["temporal_navigation"] = 0.0

        # 3. Anomaly onset: did agent inspect the anomaly_start_frame?
        onset_key_found = False
        for key in state.frames_inspected:
            _, idx_str = key.split(":")
            if int(idx_str) == gt.anomaly_start_frame:
                onset_key_found = True
                break
        if onset_key_found:
            breakdown["anomaly_onset"] = 0.20
            score += 0.20
        else:
            breakdown["anomaly_onset"] = 0.0

        # 4. Risk classification
        risk_payloads = self._payloads_for_action(state, ActionType.CLASSIFY_RISK)
        if risk_payloads and risk_payloads[-1] == gt.correct_risk_level.value:
            breakdown["risk_classification"] = 0.25
            score += 0.25
        else:
            breakdown["risk_classification"] = 0.0

        # 5. Escalation / dismissal
        if gt.should_escalate and state.escalated:
            breakdown["escalation_decision"] = 0.25
            score += 0.25
        elif not gt.should_escalate and state.dismissed:
            breakdown["escalation_decision"] = 0.25
            score += 0.25
        else:
            breakdown["escalation_decision"] = 0.0

        final = min(score, 1.0)
        return {
            "score": round(final, 4),
            "max_score": 1.0,
            "breakdown": breakdown,
            "grader": "MediumGrader",
        }


# ---------------------------------------------------------------------------
# Hard Grader
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Scoring for multi-camera coordinated incident tasks.

    Rubric (max 1.0):
        ✓ Multiple cameras visited          0.15
        ✓ Correct target camera identified   0.15
        ✓ Temporal reasoning across feeds    0.15
        ✓ Anomaly frames inspected           0.10
        ✓ False-positive avoidance           0.10
        ✓ Correct risk classification        0.15
        ✓ Correct escalation / dismiss       0.20
    """

    def evaluate(self, gt: TaskGroundTruth, state: EpisodeState) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        # 1. Multiple cameras visited (at least 2)
        unique_cameras = set(state.cameras_visited)
        if len(unique_cameras) >= 3:
            breakdown["camera_coverage"] = 0.15
            score += 0.15
        elif len(unique_cameras) >= 2:
            breakdown["camera_coverage"] = 0.10
            score += 0.10
        else:
            breakdown["camera_coverage"] = 0.0

        # 2. Correct target camera
        if gt.correct_camera in state.cameras_visited:
            # Bonus if it was the camera where escalation happened
            if state.current_camera == gt.correct_camera:
                breakdown["target_camera"] = 0.15
                score += 0.15
            else:
                breakdown["target_camera"] = 0.10
                score += 0.10
        else:
            breakdown["target_camera"] = 0.0

        # 3. Temporal reasoning — navigated through frames on >=2 cameras
        nav_actions = [ActionType.REQUEST_PREVIOUS_FRAME, ActionType.REQUEST_NEXT_FRAME]
        nav_count = sum(1 for h in state.action_history if h["action_type"] in nav_actions)
        switch_count = sum(
            1 for h in state.action_history
            if h["action_type"] == ActionType.SWITCH_CAMERA
        )
        if nav_count >= 2 and switch_count >= 2:
            breakdown["temporal_reasoning"] = 0.15
            score += 0.15
        elif nav_count >= 1 or switch_count >= 1:
            breakdown["temporal_reasoning"] = 0.08
            score += 0.08
        else:
            breakdown["temporal_reasoning"] = 0.0

        # 4. Anomaly frames inspected
        anomaly_inspected = 0
        for key in state.frames_inspected:
            _, idx_str = key.split(":")
            idx = int(idx_str)
            if idx < len(gt.frames) and gt.frames[idx].anomaly_present:
                anomaly_inspected += 1
        if anomaly_inspected >= 2:
            breakdown["anomaly_inspection"] = 0.10
            score += 0.10
        elif anomaly_inspected == 1:
            breakdown["anomaly_inspection"] = 0.05
            score += 0.05
        else:
            breakdown["anomaly_inspection"] = 0.0

        # 5. False-positive avoidance — no false escalation of clean cameras
        false_positives = 0
        escalation_payloads = self._payloads_for_action(state, ActionType.CLASSIFY_RISK)
        non_anomaly_cameras = [
            cid for cid in gt.camera_ids if cid != gt.correct_camera
        ]
        # If agent classified risk while on a non-anomaly camera: penalty
        for h in state.action_history:
            if h["action_type"] == ActionType.CLASSIFY_RISK:
                # Check if the step was on a non-threat camera
                step_idx = h["step"]
                # We can approximate: if classified risk != correct, it's suspect
                if h.get("payload") and h["payload"] != gt.correct_risk_level.value:
                    false_positives += 1

        if false_positives == 0:
            breakdown["false_positive_avoidance"] = 0.10
            score += 0.10
        else:
            breakdown["false_positive_avoidance"] = max(0.0, 0.10 - 0.05 * false_positives)
            score += breakdown["false_positive_avoidance"]

        # 6. Risk classification
        risk_payloads = self._payloads_for_action(state, ActionType.CLASSIFY_RISK)
        if risk_payloads and risk_payloads[-1] == gt.correct_risk_level.value:
            breakdown["risk_classification"] = 0.15
            score += 0.15
        else:
            breakdown["risk_classification"] = 0.0

        # 7. Correct escalation / dismissal
        if gt.should_escalate and state.escalated:
            breakdown["escalation_decision"] = 0.20
            score += 0.20
        elif not gt.should_escalate and state.dismissed:
            breakdown["escalation_decision"] = 0.20
            score += 0.20
        else:
            breakdown["escalation_decision"] = 0.0

        final = min(score, 1.0)
        return {
            "score": round(final, 4),
            "max_score": 1.0,
            "breakdown": breakdown,
            "grader": "HardGrader",
        }


# ---------------------------------------------------------------------------
# Grader Factory / Dispatcher
# ---------------------------------------------------------------------------

_GRADER_REGISTRY: Dict[TaskDifficulty, BaseGrader] = {
    TaskDifficulty.EASY: EasyGrader(),
    TaskDifficulty.MEDIUM: MediumGrader(),
    TaskDifficulty.HARD: HardGrader(),
}


def grade(
    gt: TaskGroundTruth,
    state: EpisodeState,
) -> Dict[str, Any]:
    """
    Top-level grading entry point.

    Selects the correct strategy grader based on task difficulty and
    evaluates the agent's action trajectory.

    Returns:
        dict containing ``score``, ``breakdown``, ``grader``, and summary fields.
    """
    grader = _GRADER_REGISTRY.get(gt.difficulty)
    if grader is None:
        raise ValueError(f"No grader registered for difficulty '{gt.difficulty}'")

    result = grader.evaluate(gt, state)
    result["task_id"] = gt.task_id
    result["difficulty"] = gt.difficulty.value
    result["steps_taken"] = state.current_step
    result["optimal_steps"] = gt.optimal_steps
    result["cumulative_env_reward"] = state.cumulative_reward

    logger.info(
        "Graded task=%s  score=%.3f  steps=%d/%d  grader=%s",
        gt.task_id,
        result["score"],
        state.current_step,
        gt.max_steps,
        result["grader"],
    )
    return result
