<div align="center">

# 🛡️ SentinelOps
**Interactive Agentic AI Control Room**

*Built by **Team Adaptrix** for the Meta Llama × Scaler OpenEnv Hackathon*

---

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-34E27A?style=for-the-badge)](https://github.com/huggingface/openenv)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=Pydantic&logoColor=white)](https://docs.pydantic.dev/)

</div>

## 📌 Short Project Description

**SentinelOps** is a highly dynamic, multi-step RL evaluation environment simulating a real-world CCTV surveillance control room. It challenges Vision-Language Models (VLMs) and LLM-based autonomous agents to actively manage live incident response. 

Unlike traditional static benchmarks (where models just answer questions about an image), SentinelOps drops agents into an interactive timeline. Agents must navigate 1–4 camera feeds across the facility, zoom into precise regions to crop suspect details, determine risk levels across temporal dimensions, make an escalation/dismissal judgment, and avoid "dummy flow" logic traps.

SentinelOps enforces real-world physics into its state transitions: switching cameras forces timeline desync resolution, zooming genuinely crops base64 images rendering textual hints useless, and temporal timeline scrolling is strictly confined to individual security cameras.

## 👥 Team Adaptrix
- **Project Team:** Tanish Garg, Udit Jain, Ananya Soni
- **Hackathon:** Meta Llama × Scaler OpenEnv Hackathon 2026

## 🚀 Key Features

* **12 Comprehensive Tasks**: 4 Easy, 4 Medium, 4 Hard trajectories (spanning simple parking false-alarms to coordinated multi-camera warehouse theft).
* **True Visual Reasoning & Zoom Cropping**: Real base64 security frames are fed to agents. The `zoom_region` action mathematically crops the exact physical image region matrix and halts textual "scene description" hints to prevent agents from cheating.
* **Complex Multi-Camera Tracking**: Agents navigate up to 4 parallel temporal camera feeds simultaneously.
* **Deterministic Rubric Grader**: Full adherence to the hackathon's "No LLM-as-a-judge" rule. Uses rigorous trajectory inspection yielding a stable, float `(0.0-1.0)` score based on false-positive rates, action-spam penalties, timeline pathfinding correctness, and resolution speed.
* **Streamlit Visualizer UI**: A completely custom `ui.py` graphical test-harness showing HUD layouts, time context metadata, dynamic action payloads, and live grading.
* **MCP / OpenEnv API Compliance**: Fully integrates natively as an HF Space (`/`, `/reset`, `/step`, `/grade`, `/mcp`, `/health`, `/metrics`, `/tasks`, `/schema`, `/metadata`).
* **Bulletproof Testing**: Supported by a 65/65 test suite execution path (`tests/`).

## 🛠️ How to Run Locally

You can test the actual RL API and the interactive GUI separately. 

### Prerequisites
Make sure you have Python 3.11+ installed.
```bash
git clone https://github.com/tnshgarg/sentinel-ops.git
cd sentinel-ops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Run the FastAPI Backend & HF Space Dashboard
This serves both the JSON API needed for the CLI agents AND a polished HTML control-room themed dashboard exactly like it appears on HuggingFace Spaces.
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
- **Dashboard:** [http://localhost:7860](http://localhost:7860) 
- **API Docs:** [http://localhost:7860/docs](http://localhost:7860/docs)

### 2. Run the Interactive Streamlit Visualizer UI
Want to manually test out the RL environment features (the Zoom functionality, jumping cameras, timeline sliding) from the perspective of an Agent?
```bash
python -m streamlit run ui.py --server.port 8501
```
- **Visualizer GUI:** [http://localhost:8501](http://localhost:8501)

### 3. Run the LLM Baseline (Autonomous Mode)
Run a custom fallback/smart deterministic test agent looping cleanly over the exact hackathon submission environment utilizing frontier LLMs.
```bash
# Set your HF Hub key:
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"

# Execute the agent runner against the active local port
python inference.py
```

### 4. Running the Test Suite
SentinelOps ships with exactly 65 extremely rigorous test validations.
```bash
python -m pytest tests/ -v
```

---

## 🏆 Scoring Schema & Grading Mechanics
The deterministic evaluation system works seamlessly per the hackathon requirements. 
* **(25%) Correct Anomaly Detection**: Does the agent correctly flag the event early?
* **(25%) Temporal Spatial Nav**: Do they utilize `request_next_frame`, `request_previous_frame` strictly logically or do they spam-actions for points?
* **(30%) Classification & Escalation correctness**: Did they escalate a scheduled maintenance (False Positive Penalty) or did they safely resolve a threat (Score Boost)?
* **(20%) Operational Efficiency Speed**: Solved optimally within `{optimum_steps}` yields a high bonus. Negative points awarded for repeating equivalent states without reasoning.
