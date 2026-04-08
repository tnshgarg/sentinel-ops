"""
SentinelOps — Grader Unit Tests.

Validates the deterministic rubric scoring for all difficulty tiers:
  • EasyGrader   — single-camera anomaly detection
  • MediumGrader — multi-frame temporal reasoning
  • HardGrader   — multi-camera coordinated incidents
  • grade()      — auto-dispatch to correct strategy
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grader import grade, EasyGrader, MediumGrader, HardGrader
from models import (
    ActionType,
    EpisodeState,
    RiskLevel,
    TaskDifficulty,
    TaskGroundTruth,
    FrameAnnotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gt(
    difficulty: TaskDifficulty = TaskDifficulty.EASY,
    correct_risk: RiskLevel = RiskLevel.DANGEROUS,
    should_escalate: bool = True,
    camera_ids: list | None = None,
    correct_camera: str = "cam-01",
    total_frames: int = 3,
    anomaly_start: int = 1,
) -> TaskGroundTruth:
    """Build a minimal ground-truth for testing."""
    frames = []
    for i in range(total_frames):
        frames.append(FrameAnnotation(
            frame_id=f"test-f{i}",
            camera_id=correct_camera if camera_ids is None else camera_ids[i % len(camera_ids)],
            description=f"Frame {i}",
            anomaly_present=(i >= anomaly_start),
            anomaly_type="test-anomaly" if i >= anomaly_start else None,
            anomaly_region="center" if i >= anomaly_start else None,
        ))
    return TaskGroundTruth(
        task_id="test-001",
        difficulty=difficulty,
        title="Test Task",
        description="Test",
        camera_ids=camera_ids or ["cam-01"],
        correct_camera=correct_camera,
        correct_risk_level=correct_risk,
        should_escalate=should_escalate,
        anomaly_start_frame=anomaly_start,
        total_frames=total_frames,
        frames=frames,
        optimal_steps=3,
        max_steps=10,
    )


def _make_state(
    task_id: str = "test-001",
    actions: list | None = None,
    risk_classified: str | None = None,
    escalated: bool = False,
    dismissed: bool = False,
    frames_inspected: list | None = None,
    cameras_visited: list | None = None,
    current_camera: str = "cam-01",
    current_step: int = 5,
) -> EpisodeState:
    """Build a minimal episode state for testing."""
    history = []
    if actions:
        for i, (at, payload) in enumerate(actions):
            history.append({
                "step": i,
                "action_type": at,
                "payload": payload,
                "confidence": 1.0,
                "reward": 0.0,
            })
    return EpisodeState(
        task_id=task_id,
        current_step=current_step,
        current_camera=current_camera,
        action_history=history,
        risk_classified=risk_classified,
        escalated=escalated,
        dismissed=dismissed,
        frames_inspected=frames_inspected or [],
        cameras_visited=cameras_visited or [current_camera],
        done=True,
    )


# ---------------------------------------------------------------------------
# Easy Grader Tests
# ---------------------------------------------------------------------------

class TestEasyGrader:

    def test_perfect_score(self):
        gt = _make_gt(TaskDifficulty.EASY, RiskLevel.DANGEROUS, True)
        state = _make_state(
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.CLASSIFY_RISK, "dangerous"),
                (ActionType.ESCALATE_INCIDENT, None),
            ],
            risk_classified="dangerous",
            escalated=True,
            frames_inspected=["cam-01:1"],  # anomaly is at frame 1
            current_step=3,
        )
        result = grade(gt, state)
        assert result["score"] >= 0.99
        assert result["grader"] == "EasyGrader"

    def test_zero_score(self):
        gt = _make_gt(TaskDifficulty.EASY, RiskLevel.DANGEROUS, True)
        state = _make_state(actions=[], frames_inspected=[])
        result = grade(gt, state)
        assert result["score"] <= 0.01

    def test_partial_score_no_escalation(self):
        gt = _make_gt(TaskDifficulty.EASY, RiskLevel.DANGEROUS, True)
        state = _make_state(
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
            ],
            frames_inspected=["cam-01:1"],
        )
        result = grade(gt, state)
        assert 0.0 < result["score"] < 1.0

    def test_wrong_risk_classification(self):
        gt = _make_gt(TaskDifficulty.EASY, RiskLevel.DANGEROUS, True)
        state = _make_state(
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.CLASSIFY_RISK, "safe"),
                (ActionType.ESCALATE_INCIDENT, None),
            ],
            risk_classified="safe",
            escalated=True,
            frames_inspected=["cam-01:1"],
            current_step=3,
        )
        result = grade(gt, state)
        # Should get points for inspection, detection, and decision, but not risk
        assert result["breakdown"].get("correctness_risk", 0.0) == 0.0
        assert result["breakdown"].get("correctness_decision", 0.0) == 0.30

    def test_correct_dismissal(self):
        gt = _make_gt(TaskDifficulty.EASY, RiskLevel.SAFE, False)
        state = _make_state(
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.DISMISS_ALERT, None),
            ],
            dismissed=True,
            frames_inspected=["cam-01:0"],
            current_step=2,
        )
        result = grade(gt, state)
        assert result["breakdown"]["correctness_decision"] == 0.30


# ---------------------------------------------------------------------------
# Medium Grader Tests
# ---------------------------------------------------------------------------

class TestMediumGrader:

    def test_perfect_score(self):
        gt = _make_gt(TaskDifficulty.MEDIUM, RiskLevel.DANGEROUS, True, anomaly_start=2, total_frames=4)
        state = _make_state(
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.REQUEST_NEXT_FRAME, None),
                (ActionType.REQUEST_NEXT_FRAME, None),
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.CLASSIFY_RISK, "dangerous"),
                (ActionType.ESCALATE_INCIDENT, None),
            ],
            risk_classified="dangerous",
            escalated=True,
            frames_inspected=["cam-01:0", "cam-01:2"],
            current_step=3,
        )
        result = grade(gt, state)
        assert result["score"] >= 0.99
        assert result["grader"] == "MediumGrader"

    def test_no_temporal_navigation(self):
        gt = _make_gt(TaskDifficulty.MEDIUM, RiskLevel.DANGEROUS, True, anomaly_start=2, total_frames=4)
        state = _make_state(
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.CLASSIFY_RISK, "dangerous"),
                (ActionType.ESCALATE_INCIDENT, None),
            ],
            risk_classified="dangerous",
            escalated=True,
            frames_inspected=["cam-01:0"],
            current_step=3,
        )
        result = grade(gt, state)
        assert result["breakdown"].get("efficiency_nav", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Hard Grader Tests
# ---------------------------------------------------------------------------

class TestHardGrader:

    def test_perfect_multi_camera(self):
        gt = _make_gt(
            TaskDifficulty.HARD,
            RiskLevel.CRITICAL,
            True,
            camera_ids=["cam-01", "cam-02", "cam-03", "cam-04"],
            correct_camera="cam-02",
            total_frames=6,
            anomaly_start=3,
        )
        state = _make_state(
            current_camera="cam-02",
            cameras_visited=["cam-04", "cam-03", "cam-02"],
            actions=[
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.SWITCH_CAMERA, "cam-03"),
                (ActionType.REQUEST_NEXT_FRAME, None),
                (ActionType.SWITCH_CAMERA, "cam-02"),
                (ActionType.REQUEST_NEXT_FRAME, None),
                (ActionType.INSPECT_CURRENT_FRAME, None),
                (ActionType.CLASSIFY_RISK, "critical"),
                (ActionType.ESCALATE_INCIDENT, None),
            ],
            risk_classified="critical",
            escalated=True,
            frames_inspected=["cam-04:0", "cam-02:3", "cam-02:5"],
            current_step=3,
        )
        result = grade(gt, state)
        assert result["score"] >= 0.99
        assert result["grader"] == "HardGrader"

    def test_single_camera_partial(self):
        gt = _make_gt(
            TaskDifficulty.HARD,
            RiskLevel.CRITICAL,
            True,
            camera_ids=["cam-01", "cam-02", "cam-03"],
            correct_camera="cam-02",
        )
        state = _make_state(
            cameras_visited=["cam-01"],
            current_camera="cam-01",
            actions=[
                (ActionType.ESCALATE_INCIDENT, None),
            ],
            escalated=True,
        )
        result = grade(gt, state)
        # Should get escalation points but penalised for poor camera coverage
        assert result["score"] < 0.5


# ---------------------------------------------------------------------------
# Grade Dispatcher Tests
# ---------------------------------------------------------------------------

class TestGradeDispatcher:

    def test_auto_selects_easy_grader(self):
        gt = _make_gt(TaskDifficulty.EASY)
        state = _make_state()
        result = grade(gt, state)
        assert result["grader"] == "EasyGrader"

    def test_auto_selects_medium_grader(self):
        gt = _make_gt(TaskDifficulty.MEDIUM)
        state = _make_state()
        result = grade(gt, state)
        assert result["grader"] == "MediumGrader"

    def test_auto_selects_hard_grader(self):
        gt = _make_gt(TaskDifficulty.HARD, camera_ids=["cam-01", "cam-02"])
        state = _make_state()
        result = grade(gt, state)
        assert result["grader"] == "HardGrader"

    def test_result_contains_metadata(self):
        gt = _make_gt(TaskDifficulty.EASY)
        state = _make_state()
        result = grade(gt, state)
        assert "task_id" in result
        assert "difficulty" in result
        assert "steps_taken" in result
        assert "optimal_steps" in result
        assert "score" in result
        assert "breakdown" in result
