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

    @staticmethod
    def _calculate_efficiency_bonus(state: EpisodeState, gt: TaskGroundTruth) -> float:
        """Bonus for completing the task within optimal steps."""
        if state.done and state.current_step <= gt.optimal_steps:
            return 0.10  # 10% efficiency bonus
        return 0.0

    @staticmethod
    def _calculate_safety_penalty(state: EpisodeState) -> float:
        """Penalty for erratic behavior (spamming actions)."""
        # Over 3 consecutive identical actions
        penalty = 0.0
        if len(state.action_history) >= 4:
            for i in range(len(state.action_history) - 3):
                subset = [h["action_type"] for h in state.action_history[i:i+4]]
                if len(set(subset)) == 1:
                    penalty -= 0.05
        return max(-0.15, penalty)


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

        # 1. Correctness (Sparse) - 0.70 total
        # Did the agent inspect the right frame?
        anomaly_detected = False
        for key in state.frames_inspected:
            cam, idx_str = key.split(":")
            idx = int(idx_str)
            if idx < len(gt.frames) and gt.frames[idx].anomaly_present:
                anomaly_detected = True
                break
        
        if anomaly_detected:
            breakdown["correctness_detection"] = 0.20
            score += 0.20
        
        # Correct risk classification
        risk_payloads = [h.get("payload") for h in state.action_history if h["action_type"] == ActionType.CLASSIFY_RISK]
        if risk_payloads and risk_payloads[-1] == gt.correct_risk_level.value:
            breakdown["correctness_risk"] = 0.30
            score += 0.30
            
        # Final decision
        if (gt.should_escalate and state.escalated) or (not gt.should_escalate and state.dismissed):
            breakdown["correctness_decision"] = 0.30
            score += 0.30

        # 2. Efficiency (Dense) - 0.20 total
        # Points for navigation and optimal steps
        if self._has_action(state, ActionType.INSPECT_CURRENT_FRAME):
            breakdown["efficiency_protocol"] = 0.10
            score += 0.10
        
        eff_bonus = self._calculate_efficiency_bonus(state, gt)
        breakdown["efficiency_speed"] = eff_bonus
        score += eff_bonus

        # 3. Safety (Penalties)
        safety = self._calculate_safety_penalty(state)
        breakdown["safety_compliance"] = safety
        score += safety

        final = max(0.0, min(score, 1.0))
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

        # 1. Correctness (Sparse) - 0.60
        # Anomaly onset identification
        onset_found = any(int(key.split(":")[1]) == gt.anomaly_start_frame for key in state.frames_inspected)
        if onset_found:
            breakdown["correctness_onset"] = 0.20
            score += 0.20
        
        # Risk & Decision
        risk_payloads = [h.get("payload") for h in state.action_history if h["action_type"] == ActionType.CLASSIFY_RISK]
        if risk_payloads and risk_payloads[-1] == gt.correct_risk_level.value:
            breakdown["correctness_risk"] = 0.25
            score += 0.25
            
        if (gt.should_escalate and state.escalated) or (not gt.should_escalate and state.dismissed):
            breakdown["correctness_decision"] = 0.25
            score += 0.25

        # 2. Efficiency (Dense) - 0.30
        nav_actions = [ActionType.REQUEST_PREVIOUS_FRAME, ActionType.REQUEST_NEXT_FRAME]
        nav_count = sum(1 for h in state.action_history if h["action_type"] in nav_actions)
        if nav_count >= 2:
            breakdown["efficiency_nav"] = 0.10
            score += 0.10
        
        eff_bonus = self._calculate_efficiency_bonus(state, gt)
        breakdown["efficiency_speed"] = eff_bonus * 2  # Harder tasks get more speed reward
        score += breakdown["efficiency_speed"]

        # 3. Safety
        safety = self._calculate_safety_penalty(state)
        breakdown["safety_compliance"] = safety
        score += safety

        final = max(0.0, min(score, 1.0))
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

    New Multi-Layer Rubric (max 1.0):
        1. Correctness (Sparse) - Phase-based accuracy (Detection, Risk, Outcome)
        2. Efficiency (Dense) - Navigation and optimal camera switching
        3. Safety (Penalty) - Strategic attention (no action spamming)
    """

    def evaluate(self, gt: TaskGroundTruth, state: EpisodeState) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        # 1. Correctness (Sparse) - 0.50
        # Multi-camera tracking
        unique_cameras = set(state.cameras_visited)
        if gt.correct_camera in unique_cameras:
            breakdown["correctness_coverage"] = 0.15
            score += 0.15
        
        # Anomaly inspection depth
        anomaly_inspected = sum(1 for key in state.frames_inspected if gt.frames[int(key.split(":")[1])].anomaly_present)
        if anomaly_inspected >= 2:
            breakdown["correctness_inspection"] = 0.20
            score += 0.20
            
        # Risk & Decision
        risk_payloads = [h.get("payload") for h in state.action_history if h["action_type"] == ActionType.CLASSIFY_RISK]
        if risk_payloads and risk_payloads[-1] == gt.correct_risk_level.value:
            if (gt.should_escalate and state.escalated) or (not gt.should_escalate and state.dismissed):
                breakdown["correctness_outcome"] = 0.25
                score += 0.25

        # 2. Efficiency (Dense) - 0.40
        # Camera switching logic
        switch_count = sum(1 for h in state.action_history if h["action_type"] == ActionType.SWITCH_CAMERA)
        if 1 <= switch_count <= len(gt.camera_ids) + 1:
            breakdown["efficiency_switching"] = 0.15
            score += 0.15
            
        # Navigation logic
        nav_actions = [ActionType.REQUEST_PREVIOUS_FRAME, ActionType.REQUEST_NEXT_FRAME]
        nav_count = sum(1 for h in state.action_history if h["action_type"] in nav_actions)
        if nav_count >= 2:
            breakdown["efficiency_navigation"] = 0.15
            score += 0.15
            
        eff_bonus = self._calculate_efficiency_bonus(state, gt)
        breakdown["efficiency_speed"] = eff_bonus
        score += eff_bonus

        # 3. Safety (Penalties)
        safety = self._calculate_safety_penalty(state)
        breakdown["safety_compliance"] = safety
        score += safety

        final = max(0.0, min(score, 1.0))
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
