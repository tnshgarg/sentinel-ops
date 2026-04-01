"""
SentinelOps — Environment Unit Tests.

Validates the complete environment lifecycle:
  • reset()  returns valid observations
  • step()   transitions state correctly
  • all action types are handled
  • termination conditions work
  • edge cases (double-reset, step-after-done, boundaries)
"""

from __future__ import annotations

import pytest

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import SentinelOpsEnvironment, TaskFactory
from models import Action, ActionType, AlertLevel, TaskDifficulty


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh environment instance for each test."""
    return SentinelOpsEnvironment()


@pytest.fixture
def easy_task_id():
    tasks = TaskFactory.load_by_difficulty(TaskDifficulty.EASY)
    assert len(tasks) > 0, "No easy tasks found"
    return tasks[0].task_id


@pytest.fixture
def hard_task_id():
    tasks = TaskFactory.load_by_difficulty(TaskDifficulty.HARD)
    assert len(tasks) > 0, "No hard tasks found"
    return tasks[0].task_id


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_returns_observation(self, env, easy_task_id):
        obs, info = env.reset(easy_task_id)
        assert obs.task_id == easy_task_id
        assert obs.step == 0
        assert obs.camera_id is not None
        assert len(obs.frame_b64) > 0
        assert len(obs.available_actions) > 0

    def test_reset_random_task(self, env):
        obs, info = env.reset()
        assert obs.task_id is not None
        assert "task_id" in info

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent-task-999")

    def test_double_reset(self, env, easy_task_id):
        obs1, _ = env.reset(easy_task_id)
        obs2, _ = env.reset(easy_task_id)
        assert obs2.step == 0, "Reset should start fresh"

    def test_reset_info_contents(self, env, easy_task_id):
        _, info = env.reset(easy_task_id)
        assert "difficulty" in info
        assert "title" in info
        assert "total_frames" in info
        assert "max_steps" in info
        assert "camera_ids" in info


# ---------------------------------------------------------------------------
# Step Tests
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_inspect(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.INSPECT_CURRENT_FRAME)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.step == 1
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_increments_counter(self, env, easy_task_id):
        env.reset(easy_task_id)
        for i in range(3):
            action = Action(action_type=ActionType.INSPECT_CURRENT_FRAME)
            obs, *_ = env.step(action)
            assert obs.step == i + 1

    def test_step_next_frame(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.REQUEST_NEXT_FRAME)
        obs, *_ = env.step(action)
        state = env.state()
        assert state.current_frame_idx >= 0  # Should have moved forward if possible

    def test_step_previous_frame_at_start(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.REQUEST_PREVIOUS_FRAME)
        obs, *_ = env.step(action)
        state = env.state()
        assert state.current_frame_idx == 0  # Should stay at 0

    def test_step_classify_risk(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.CLASSIFY_RISK, payload="dangerous")
        obs, *_ = env.step(action)
        state = env.state()
        assert state.risk_classified == "dangerous"

    def test_step_zoom(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.ZOOM_REGION, payload="center")
        obs, *_ = env.step(action)
        state = env.state()
        assert state.zoom_active is True
        assert state.zoom_region == "center"

    def test_step_escalate_terminates(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.ESCALATE_INCIDENT)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert env.state().done is True

    def test_step_dismiss_terminates(self, env, easy_task_id):
        env.reset(easy_task_id)
        action = Action(action_type=ActionType.DISMISS_ALERT)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert env.state().done is True

    def test_step_after_done_raises(self, env, easy_task_id):
        env.reset(easy_task_id)
        env.step(Action(action_type=ActionType.ESCALATE_INCIDENT))
        with pytest.raises(RuntimeError, match="already finished"):
            env.step(Action(action_type=ActionType.INSPECT_CURRENT_FRAME))

    def test_step_without_reset_raises(self):
        fresh_env = SentinelOpsEnvironment()
        with pytest.raises(RuntimeError, match="not initialised"):
            fresh_env.step(Action(action_type=ActionType.INSPECT_CURRENT_FRAME))


# ---------------------------------------------------------------------------
# Multi-Camera Tests
# ---------------------------------------------------------------------------

class TestMultiCamera:

    def test_switch_camera(self, env, hard_task_id):
        obs, info = env.reset(hard_task_id)
        cameras = info["camera_ids"]
        assert len(cameras) > 1

        target = [c for c in cameras if c != obs.camera_id][0]
        action = Action(action_type=ActionType.SWITCH_CAMERA, payload=target)
        obs, *_ = env.step(action)
        assert env.state().current_camera == target

    def test_cameras_visited_tracking(self, env, hard_task_id):
        obs, info = env.reset(hard_task_id)
        cameras = info["camera_ids"]

        # Visit a second camera
        target = [c for c in cameras if c != obs.camera_id][0]
        env.step(Action(action_type=ActionType.SWITCH_CAMERA, payload=target))

        state = env.state()
        assert len(state.cameras_visited) >= 2


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------

class TestState:

    def test_state_returns_copy(self, env, easy_task_id):
        env.reset(easy_task_id)
        s1 = env.state()
        s2 = env.state()
        assert s1 is not s2  # Should be independent copies

    def test_state_no_episode_raises(self):
        fresh_env = SentinelOpsEnvironment()
        with pytest.raises(RuntimeError, match="No active episode"):
            fresh_env.state()

    def test_action_history_recorded(self, env, easy_task_id):
        env.reset(easy_task_id)
        env.step(Action(action_type=ActionType.INSPECT_CURRENT_FRAME))
        env.step(Action(action_type=ActionType.CLASSIFY_RISK, payload="safe"))

        state = env.state()
        assert len(state.action_history) == 2
        assert state.action_history[0]["action_type"] == ActionType.INSPECT_CURRENT_FRAME
        assert state.action_history[1]["action_type"] == ActionType.CLASSIFY_RISK


# ---------------------------------------------------------------------------
# Truncation Tests
# ---------------------------------------------------------------------------

class TestTruncation:

    def test_truncation_at_max_steps(self, env, easy_task_id):
        obs, info = env.reset(easy_task_id)
        max_steps = info["max_steps"]

        for i in range(max_steps):
            action = Action(action_type=ActionType.INSPECT_CURRENT_FRAME)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert env.state().done is True


# ---------------------------------------------------------------------------
# Task Factory Tests
# ---------------------------------------------------------------------------

class TestTaskFactory:

    def test_load_all(self):
        tasks = TaskFactory.load_all()
        assert len(tasks) >= 6, f"Expected at least 6 tasks, got {len(tasks)}"

    def test_load_by_difficulty(self):
        for diff in TaskDifficulty:
            tasks = TaskFactory.load_by_difficulty(diff)
            assert len(tasks) >= 1, f"No tasks found for difficulty {diff.value}"

    def test_load_by_id(self):
        tasks = TaskFactory.load_all()
        for task in tasks:
            loaded = TaskFactory.load_by_id(task.task_id)
            assert loaded is not None
            assert loaded.task_id == task.task_id


# ---------------------------------------------------------------------------
# Action Validation Tests
# ---------------------------------------------------------------------------

class TestActionValidation:

    def test_invalid_action_type_rejected(self):
        with pytest.raises(ValueError, match="Invalid action_type"):
            Action(action_type="fly_helicopter")

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Action(action_type=ActionType.INSPECT_CURRENT_FRAME, confidence=1.5)

    def test_all_action_types_valid(self):
        for at in ActionType:
            action = Action(action_type=at.value)
            assert action.action_type == at.value
