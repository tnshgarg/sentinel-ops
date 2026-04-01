"""
SentinelOps — API Integration Tests.

Tests the FastAPI server endpoints end-to-end using httpx TestClient.
Validates:
  • /health returns 200
  • /reset returns valid observation
  • /step processes actions correctly
  • /state returns episode state
  • /tasks lists available tasks
  • /grade returns scoring after episode completion
  • Error handling for invalid requests
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from server.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a test client that shares state across requests in a test."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & Root Tests
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "sentinelops"

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "docs" in data


# ---------------------------------------------------------------------------
# Reset Endpoint Tests
# ---------------------------------------------------------------------------

class TestResetEndpoint:

    def test_reset_without_task_id(self, client):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "info" in data
        obs = data["observation"]
        assert "task_id" in obs
        assert obs["step"] == 0
        assert len(obs["frame_b64"]) > 0

    def test_reset_with_task_id(self, client):
        # First get available tasks
        tasks_resp = client.get("/tasks")
        tasks = tasks_resp.json()
        assert len(tasks) > 0

        task_id = tasks[0]["task_id"]
        resp = client.post("/reset", json={"task_id": task_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["task_id"] == task_id

    def test_reset_invalid_task(self, client):
        resp = client.post("/reset", json={"task_id": "no-such-task"})
        assert resp.status_code == 404

    def test_reset_no_body(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Step Endpoint Tests
# ---------------------------------------------------------------------------

class TestStepEndpoint:

    def test_step_inspect(self, client):
        client.post("/reset", json={})
        resp = client.post("/step", json={
            "action_type": "inspect_current_frame",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "terminated" in data
        assert "truncated" in data
        assert data["observation"]["step"] == 1

    def test_step_escalate_terminates(self, client):
        client.post("/reset", json={})
        resp = client.post("/step", json={
            "action_type": "escalate_incident",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["terminated"] is True

    def test_step_invalid_action(self, client):
        client.post("/reset", json={})
        resp = client.post("/step", json={
            "action_type": "launch_rocket",
        })
        assert resp.status_code == 422

    def test_step_with_payload(self, client):
        client.post("/reset", json={})
        resp = client.post("/step", json={
            "action_type": "classify_risk",
            "payload": "dangerous",
            "confidence": 0.95,
        })
        assert resp.status_code == 200

    def test_step_before_reset(self, client):
        # Use a fresh app — this test depends on server state
        # Since we share state, we just reset and then test after termination
        client.post("/reset", json={})
        client.post("/step", json={"action_type": "escalate_incident"})
        resp = client.post("/step", json={"action_type": "inspect_current_frame"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# State Endpoint Tests
# ---------------------------------------------------------------------------

class TestStateEndpoint:

    def test_state_after_reset(self, client):
        client.post("/reset", json={})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "episode" in data
        assert "task_metadata" in data
        assert data["episode"]["current_step"] == 0

    def test_state_after_step(self, client):
        client.post("/reset", json={})
        client.post("/step", json={"action_type": "inspect_current_frame"})
        resp = client.get("/state")
        data = resp.json()
        assert data["episode"]["current_step"] == 1
        assert len(data["episode"]["action_history"]) == 1


# ---------------------------------------------------------------------------
# Tasks Endpoint Tests
# ---------------------------------------------------------------------------

class TestTasksEndpoint:

    def test_list_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)
        assert len(tasks) >= 12
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "title" in task


# ---------------------------------------------------------------------------
# Grade Endpoint Tests
# ---------------------------------------------------------------------------

class TestGradeEndpoint:

    def test_grade_after_completion(self, client):
        client.post("/reset", json={})
        client.post("/step", json={"action_type": "inspect_current_frame"})
        client.post("/step", json={"action_type": "classify_risk", "payload": "dangerous"})
        client.post("/step", json={"action_type": "escalate_incident"})

        resp = client.post("/grade")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "breakdown" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_grade_before_completion(self, client):
        client.post("/reset", json={})
        client.post("/step", json={"action_type": "inspect_current_frame"})
        resp = client.post("/grade")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Full Episode Integration Test
# ---------------------------------------------------------------------------

class TestFullEpisode:

    def test_optimal_easy_episode(self, client):
        """Run a complete optimal episode on an easy task."""
        tasks = client.get("/tasks").json()
        easy_tasks = [t for t in tasks if t["difficulty"] == "easy"]
        assert len(easy_tasks) > 0

        task_id = easy_tasks[0]["task_id"]

        # Reset
        reset_resp = client.post("/reset", json={"task_id": task_id}).json()
        assert reset_resp["observation"]["step"] == 0

        # Step 1: Inspect
        step1 = client.post("/step", json={"action_type": "inspect_current_frame"}).json()
        assert step1["terminated"] is False

        # Step 2: Navigate to anomaly frame
        step2 = client.post("/step", json={"action_type": "request_next_frame"}).json()

        # Step 3: Inspect anomaly frame
        step3 = client.post("/step", json={"action_type": "inspect_current_frame"}).json()

        # Step 4: Classify risk
        step4 = client.post("/step", json={
            "action_type": "classify_risk",
            "payload": "dangerous",
        }).json()

        # Step 5: Escalate
        step5 = client.post("/step", json={"action_type": "escalate_incident"}).json()
        assert step5["terminated"] is True

        # Grade
        grade_resp = client.post("/grade").json()
        assert grade_resp["score"] > 0.0
        assert "breakdown" in grade_resp
