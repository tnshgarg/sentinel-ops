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
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from config import safe_score, SCORE_LO, SCORE_HI
from models import (
    ActionType,
    EpisodeState,
    TaskDifficulty,
    TaskGroundTruth,
)

logger = logging.getLogger("sentinelops.grader")

# Removed local safe_score - now imported from config.py


# ---------------------------------------------------------------------------
# Base Grader (Strategy interface)
# ---------------------------------------------------------------------------

class BaseGrader(ABC):
    """
    Abstract grader that evaluates an agent's episode trajectory.

    Subclasses implement ``evaluate()`` to return a score strictly in (0, 1)
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
            dict with at-minimum keys ``"score"`` (float strictly in (0, 1)) and
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

    Rubric (raw max 1.0, output clamped to (0.001, 0.999)):
        Correctness (0.80):
            ✓ Anomaly detection          0.20
            ✓ Correct risk classification 0.30
            ✓ Correct escalation/dismiss  0.30
        Efficiency (0.20):
            ✓ Frame inspection protocol  0.10
            ✓ Speed bonus (optimal steps) 0.10
        Safety: penalty up to -0.15
    """

    def evaluate(self, gt: TaskGroundTruth, state: EpisodeState) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        # 1. Correctness (Sparse) - 0.80 total
        # Did the agent inspect the right frame?
        anomaly_detected = False
        if not gt.should_escalate:
            # False alarm tasks: detection credit for investigating the scene at all
            anomaly_detected = bool(state.frames_inspected)
        else:
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
        if eff_bonus > 0:
            breakdown["efficiency_speed"] = eff_bonus
        score += eff_bonus

        # 3. Safety (Penalties) — only record in breakdown if non-zero
        safety = self._calculate_safety_penalty(state)
        if safety != 0.0:
            breakdown["safety_compliance"] = safety
        score += safety

        return {
            "score": safe_score(score),
            "breakdown": breakdown,
            "grader": "EasyGrader",
        }


# ---------------------------------------------------------------------------
# Medium Grader
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Scoring for multi-frame temporal reasoning tasks.

    Rubric (raw max 1.00, output clamped to (0.001, 0.999)):
        Correctness (0.70):
            ✓ Anomaly onset identified          0.20
            ✓ Correct risk classification       0.25
            ✓ Correct escalation / dismiss      0.25
        Efficiency (0.30):
            ✓ Temporal navigation (≥2 moves)    0.10
            ✓ Speed bonus ×2 (optimal steps)    0.20
        Safety: penalty up to -0.15
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
        speed_score = eff_bonus * 2  # Harder tasks get more speed reward
        if speed_score > 0:
            breakdown["efficiency_speed"] = speed_score
        score += speed_score

        # 3. Safety — only record in breakdown if non-zero
        safety = self._calculate_safety_penalty(state)
        if safety != 0.0:
            breakdown["safety_compliance"] = safety
        score += safety

        return {
            "score": safe_score(score),
            "breakdown": breakdown,
            "grader": "MediumGrader",
        }


# ---------------------------------------------------------------------------
# Hard Grader
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Scoring for multi-camera coordinated incident tasks.

    Rubric (raw max 1.00, output clamped to (0.001, 0.999)):
        Correctness (0.60):
            ✓ Correct camera visited            0.15
            ✓ Anomaly inspection depth (≥2)     0.20
            ✓ Risk + decision outcome           0.25
        Efficiency (0.40):
            ✓ Camera switching (1–N+1 switches) 0.15
            ✓ Frame navigation (≥2 moves)       0.15
            ✓ Speed bonus (optimal steps)       0.10
        Safety: penalty up to -0.15
    """

    def evaluate(self, gt: TaskGroundTruth, state: EpisodeState) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        # 1. Correctness (Sparse) - 0.60
        # Multi-camera tracking
        unique_cameras = set(state.cameras_visited)
        if gt.correct_camera in unique_cameras:
            breakdown["correctness_coverage"] = 0.15
            score += 0.15

        # Anomaly inspection depth
        # For false alarm tasks: any frame inspection counts (investigating the scene)
        # For real threat tasks: must inspect actual anomaly frames
        if not gt.should_escalate:
            anomaly_inspected = len(state.frames_inspected)
        else:
            anomaly_inspected = sum(
                1 for key in state.frames_inspected
                if (lambda idx: idx < len(gt.frames) and gt.frames[idx].anomaly_present)(int(key.split(":")[1]))
            )
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
        if eff_bonus > 0:
            breakdown["efficiency_speed"] = eff_bonus
        score += eff_bonus

        # 3. Safety (Penalties) — only record in breakdown if non-zero
        safety = self._calculate_safety_penalty(state)
        if safety != 0.0:
            breakdown["safety_compliance"] = safety
        score += safety

        return {
            "score": safe_score(score),
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
    # Clamp cumulative_env_reward to (0,1) so validator scanning all numeric fields
    # never sees 0.0 or 1.0 — the step-reward accumulator can hit these exact boundaries.
    result["cumulative_env_reward"] = safe_score(state.cumulative_reward)

    # OpenEnv validation requires score strictly in (0, 1) — not 0.0 and not 1.0
    result["score"] = safe_score(result.get("score", 0.5))

    logger.info(
        "Graded task=%s  score=%.3f  steps=%d/%d  grader=%s",
        gt.task_id,
        result["score"],
        state.current_step,
        gt.max_steps,
        result["grader"],
    )
    return result
