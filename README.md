<div align="center">

# 🛡️ SentinelOps

### Interactive Incident Response Control Room — OpenEnv Environment

*A multi-step agentic reinforcement-learning environment where AI agents monitor surveillance camera feeds, detect anomalies, track suspects across cameras, classify threats, and escalate incidents.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-orange.svg)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)

</div>

---

## 🎯 Overview

**SentinelOps** simulates a real-world AI security analyst operating a multi-camera surveillance control room. Unlike static VQA benchmarks, SentinelOps creates **multi-step agentic episodes** where decisions have sequential consequences and rewards are accumulated across trajectories.

### Real-World Applications

| Domain | Use Case |
|--------|----------|
| 🏙️ Smart Cities | Citywide surveillance monitoring |
| 🏬 Mall Security | Retail theft prevention & incident response |
| 🏥 Hospital Safety | Emergency room monitoring |
| 🏭 Industrial Plants | Safety violation detection |
| 🚦 Traffic Control | Automated traffic incident management |

---

## 🏗️ Architecture

```
Alert Trigger → Observation State → Agent Action → Environment Transition → Reward → Next State
```

### Episode Flow (5–10 steps)

1. **Anomaly alert** arrives on a camera feed
2. Agent **inspects** the current frame
3. Agent **navigates** temporal history (previous/next frames)
4. Agent **switches cameras** to track subjects
5. Agent **classifies** threat severity
6. Agent **escalates** or **dismisses** the alert

### Design Patterns

| Pattern | Application |
|---------|-------------|
| **Strategy** | Difficulty-specific graders (Easy/Medium/Hard) |
| **Factory** | Task loading and caching |
| **State** | Episode state machine transitions |
| **SOLID** | Single responsibility, dependency inversion |

---

## 📁 Project Structure

```
sentinelops/
├── inference.py           # Root-level inference script (submission entry point)
├── env.py                 # Core environment engine (reset/step/state)
├── grader.py              # Deterministic rubric-based graders
├── models.py              # Pydantic data models
├── config.py              # Centralised configuration
├── openenv.yaml           # OpenEnv manifest
├── requirements.txt       # Python dependencies
├── Dockerfile             # Production Docker image
├── validate-submission.sh # Pre-submission validation
├── .env.example           # Environment variable template
│
├── server/
│   └── app.py             # FastAPI HTTP server
│
├── tasks/
│   ├── easy/              # Single-camera anomaly detection
│   ├── medium/            # Multi-frame temporal reasoning
│   └── hard/              # Multi-camera coordinated incidents
│
├── assets/
│   ├── frames/            # CCTV surveillance frame images
│   └── sequences/         # Multi-frame sequences
│
└── tests/
    ├── test_env.py        # Environment unit tests
    ├── test_grader.py     # Grader unit tests
    └── test_api.py        # API integration tests
```

---

## 🎮 Action Space

| Action | Description |
|--------|-------------|
| `inspect_current_frame` | Examine the current camera frame |
| `request_previous_frame` | Navigate to the previous frame in timeline |
| `request_next_frame` | Navigate to the next frame in timeline |
| `switch_camera` | Switch to a different camera feed |
| `zoom_region` | Zoom into a specific region of the frame |
| `classify_risk` | Classify threat level (safe/suspicious/dangerous/critical) |
| `escalate_incident` | Trigger emergency response |
| `dismiss_alert` | Dismiss as false alarm |

---

## 🏆 Reward Engineering

Rewards are **accumulated across the trajectory** (delayed reward), making this significantly stronger than one-shot scoring.

| Signal | Reward |
|--------|--------|
| ✅ Correct anomaly detection | +0.20 |
| ✅ Correct temporal reasoning | +0.20 |
| ✅ Correct escalation | +0.30 |
| ⚡ Fast response (within optimal steps) | +0.10 |
| ❌ False positive | -0.20 |
| ❌ Missed anomaly | -0.40 |
| ⚠️ Random action spam | -0.10 |
| ☠️ Unsafe dismissal | -0.30 |

---

## 📊 Task Difficulty Ladder

### 🟢 Easy — Single Camera Anomaly
- Detect suspicious objects
- Identify count mismatches
- **2 tasks**, 3–8 steps

### 🟡 Medium — Multi-Frame Progression
- Identify when events begin (temporal reasoning)
- Classify movement patterns
- **2 tasks**, 5–10 steps

### 🔴 Hard — Multi-Camera Coordinated Incidents
- Switch between camera feeds
- Track subject paths across areas
- Predict threat escalation
- Decide emergency response
- **2 tasks**, 7–15 steps

---

## 📈 Baseline Results

The `inference.py` script executes a full deterministic baseline using a heuristic-based agent when LLM providers are rate-limited or unavailable. This proves the environment yields a fully scored, deterministic evaluation trajectory that never crashes.

**Execution Configuration:**
- Inference Time: **under 30 seconds** (for 12 episodes)
- Environment Setup: Local FastAPI Docker Container
- Scoring Method: Rubric-based Deterministic Evaluation
- Number of Errors/Crashes: **Zero**

### Summary
- **Total tasks:** 12
- **Average score (all tiers):** 0.750
- **Total cumulative score:** 9.0

### Per-Task Breakdown
| Task ID | Tier | Score | Steps Taken | Time (s) |
|---|---|---|---|---|
| `easy-001-parking-intrusion` | Easy | **0.70** | 5 | 2.78 |
| `easy-002-warehouse-access` | Easy | **0.70** | 5 | 2.33 |
| `easy-007-atm-tampering` | Easy | **1.00** | 5 | 2.76 |
| `easy-008-night-patrol-false-alarm` | Easy | **0.75** | 5 | 2.01 |
| `medium-003-corridor-intrusion` | Medium | **0.75** | 7 | 1.76 |
| `medium-004-lobby-surveillance` | Medium | **0.75** | 7 | 1.79 |
| `medium-009-rooftop-sabotage` | Medium | **0.75** | 7 | 1.74 |
| `medium-010-warehouse-progression` | Medium | **0.75** | 7 | 2.30 |
| `hard-005-multi-camera-pursuit` | Hard | **0.75** | 11 | 1.87 |
| `hard-006-false-alarm-discrimination` | Hard | **0.50** | 11 | 2.03 |
| `hard-011-coordinated-theft` | Hard | **0.75** | 11 | 1.86 |
| `hard-012-authorized-access-false-alarm` | Hard | **0.85** | 11 | 2.02 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for deployment)
- Hugging Face API token

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd sentinelops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Running Inference

```bash
# Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.2-11B-Vision-Instruct
export HF_TOKEN=hf_your_token_here
export ENV_URL=http://localhost:7860

# Run inference
python inference.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

### Docker

```bash
# Build
docker build -t sentinelops .

# Run
docker run -p 7860:7860 sentinelops

# Validate
./validate-submission.sh http://localhost:7860
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start/restart an episode |
| `POST` | `/step` | Submit an agent action |
| `GET` | `/state` | Get current episode state |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/grade` | Grade a completed episode |
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive API documentation |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy-001-parking-intrusion"}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_current_frame"}'
```

---

## 📐 Grading

Scoring is **fully deterministic** — no LLM-as-judge. Each difficulty tier has its own rubric:

### Easy Rubric (max 1.0)
- Frame inspection: 0.15
- Anomaly detection: 0.25
- Risk classification: 0.30
- Escalation decision: 0.30

### Medium Rubric (max 1.0)
- Frame inspection: 0.10
- Temporal navigation: 0.20
- Anomaly onset identification: 0.20
- Risk classification: 0.25
- Escalation decision: 0.25

### Hard Rubric (max 1.0)
- Camera coverage: 0.15
- Target camera identification: 0.15
- Cross-camera temporal reasoning: 0.15
- Anomaly frame inspection: 0.10
- False-positive avoidance: 0.10
- Risk classification: 0.15
- Escalation decision: 0.20

---

## ✅ Quality Gate Checklist

- [x] `openenv validate` passes
- [x] `docker build` succeeds
- [x] `docker run` starts correctly
- [x] HF Space `/reset` returns HTTP 200
- [x] Inference completes under 20 minutes
- [x] Reproducible score variance < 2%
- [x] Unit test coverage > 90%
- [x] No placeholder functions or TODOs
- [x] All endpoints runnable
- [x] Deterministic grading

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

**Built for the Meta × Scaler OpenEnv Hackathon by Team Adaptrix**

*SentinelOps — Where AI meets real-world security surveillance*

</div>
