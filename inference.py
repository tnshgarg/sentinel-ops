#!/usr/bin/env python3
"""
SentinelOps — Inference Script (Submission Entry Point).

This is the root-level inference script required by the OpenEnv hackathon spec.
It uses the OpenAI-compatible client to drive an LLM agent through
SentinelOps episodes, printing reproducible scores.

Required environment variables:
    API_BASE_URL   – base URL for the inference API
    MODEL_NAME     – model identifier
    HF_TOKEN       – Hugging Face API token

Constraints:
    • Uses ``from openai import OpenAI`` as mandated
    • Deterministic low-temperature inference (temperature=0.1)
    • Completes full episode < 20 minutes
    • Prints reproducible scores

Features:
    • Vision-Language Model support (sends base64 frames to VLMs)
    • Model fallback chain: tries primary → smaller VLMs on 402/429
    • Deterministic baseline agent when all LLM providers are exhausted
    • Graceful error handling — never crashes mid-episode
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load .env file & Configuration from environment
# ---------------------------------------------------------------------------

load_dotenv()  # Load .env file so HF_TOKEN etc. are available via os.environ

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

TEMPERATURE = 0.1
MAX_RETRIES = 2
TIMEOUT_MINUTES = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinelops.inference")


# ---------------------------------------------------------------------------
# Model Fallback Chain — VLMs only (can process base64 images)
# ---------------------------------------------------------------------------

MODEL_FALLBACK_CHAIN: List[str] = [
    MODEL_NAME,
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# De-duplicate while preserving order
_seen = set()
MODEL_CHAIN: List[str] = []
for m in MODEL_FALLBACK_CHAIN:
    if m not in _seen:
        _seen.add(m)
        MODEL_CHAIN.append(m)


# ---------------------------------------------------------------------------
# OpenAI Client Initialisation
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


# ---------------------------------------------------------------------------
# Environment HTTP Client
# ---------------------------------------------------------------------------

import requests


class EnvClient:
    """Thin wrapper around the SentinelOps environment server HTTP API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"task_id": task_id} if task_id else {}
        resp = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type: str, payload: Optional[str] = None, confidence: float = 1.0, 
             predicted_gaze: Optional[List[int]] = None, 
             velocity_vector: Optional[List[int]] = None) -> Dict[str, Any]:
        body = {
            "action_type": action_type,
            "payload": payload,
            "confidence": confidence,
            "predicted_gaze": predicted_gaze,
            "velocity_vector": velocity_vector
        }
        resp = self.session.post(f"{self.base_url}/step", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> Dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/grade", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> List[Dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/tasks", timeout=30)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are SentinelOps Agent — an elite AI security analyst operating a multi-camera surveillance control room.

Your mission is to:
1. Inspect surveillance frames for anomalies (examine the actual image carefully)
2. Navigate between frames and cameras to track threats
3. Classify the risk level of any detected anomaly
4. Decide whether to escalate the incident or dismiss the alert

ELITE SPATIAL GROUNDING (MANDATORY):
To prove your spatial internalization, you MUST include the [y, x] center-point of any detected anomaly in your reasoning. Use a normalized 0-1000 coordinate system.
Format: "[y, x]" where y is vertical and x is horizontal.

AVAILABLE ACTIONS (you must respond with EXACTLY one per turn):
- inspect_current_frame          → Examine the current camera frame closely
- request_previous_frame         → Go to the previous frame in the timeline
- request_next_frame             → Go to the next frame in the timeline
- switch_camera:<camera_id>      → Switch to a different camera feed (e.g. switch_camera:cam-02)
- zoom_region:<region>           → Zoom into a region (top-left, top-right, center, bottom-left, bottom-right, left, right)
- classify_risk:<level>          → Classify the threat level (safe, suspicious, dangerous, critical)
- escalate_incident              → Trigger emergency response for a confirmed threat
- dismiss_alert                  → Dismiss as false alarm (only if truly safe)

RESPONSE FORMAT — respond with a JSON object on a single line:
{"action": "<action_name>", "payload": "<optional_payload>", "reasoning": "<brief explanation with [y, x] coordinate>"}

Example with Grounding:
{"action": "inspect_current_frame", "payload": null, "reasoning": "Detected person near the vault at [450, 780]. Evidence suggests suspicious activity."}

DECISION RULES (follow strictly):
1. Always inspect the current frame FIRST on every new camera or after navigating.
2. In multi-camera tasks, switch cameras to track the subject before classifying risk.
3. Once you have inspected at least one anomaly frame, classify_risk IMMEDIATELY.
4. After classifying risk, your VERY NEXT action must be either escalate_incident or dismiss_alert.
5. Be decisive — fewer steps earn a speed bonus.
"""


# ---------------------------------------------------------------------------
# Deterministic Baseline Agent (used when LLM API is unavailable)
# ---------------------------------------------------------------------------

class DeterministicBaselineAgent:
    """
    A scripted agent that executes an intelligent action sequence
    based on task difficulty. Guarantees the episode completes
    without crashing and demonstrates the environment's scoring.

    This is NOT a random agent — it follows the optimal strategy
    pattern for each difficulty tier.
    """

    def __init__(self):
        self._step_index = 0
        self._difficulty = "easy"
        self._camera_ids: List[str] = []
        self._current_camera_idx = 0

    def reset(self, info: Dict[str, Any]):
        self._step_index = 0
        self._difficulty = info.get("difficulty", "easy")
        self._camera_ids = info.get("camera_ids", ["cam-01"])
        self._current_camera_idx = 0

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return the next action based on a pre-planned trajectory."""
        available = obs.get("available_actions", [])
        context = obs.get("context", "").lower()

        # Determine risk level from context clues
        risk = "suspicious"
        if any(w in context for w in ["breaking", "tamper", "theft", "weapon", "tools", "lock"]):
            risk = "dangerous"
        if any(w in context for w in ["coordinated", "armed", "critical"]):
            risk = "critical"
        if any(w in context for w in ["authorised", "authorized", "maintenance", "scheduled", "badge"]):
            risk = "safe"

        # Determine whether to escalate or dismiss
        should_escalate = risk not in ("safe",)

        if self._difficulty == "easy":
            plan = self._easy_plan(risk, should_escalate)
        elif self._difficulty == "medium":
            plan = self._medium_plan(risk, should_escalate, available)
        else:
            plan = self._hard_plan(risk, should_escalate, available)

        if self._step_index < len(plan):
            action = plan[self._step_index]
        else:
            # Safety net: force termination
            action = {"action": "escalate_incident", "payload": None, "reasoning": "Forcing episode end."}

        self._step_index += 1

        # Validate action is available
        action_type = action["action"]
        if action_type not in available and available:
            # Pick first available non-inspect action to avoid duplicate
            for fallback in ["request_next_frame", "classify_risk", "escalate_incident"]:
                if fallback in available:
                    action = {"action": fallback, "payload": action.get("payload"), "reasoning": "Fallback to available action."}
                    break

        return action

    def _easy_plan(self, risk: str, escalate: bool) -> List[Dict]:
        terminal = "escalate_incident" if escalate else "dismiss_alert"
        # Include [y, x] in baseline reasoning to support the tactical HUD
        return [
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Examining alert frame at [500, 500]."},
            {"action": "request_next_frame", "payload": None, "reasoning": "Checking temporal progression at [510, 490]."},
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Inspecting next frame for anomaly at [515, 485]."},
            {"action": "classify_risk", "payload": risk, "reasoning": f"Scene indicates {risk} threat level at [520, 480]."},
            {"action": terminal, "payload": None, "reasoning": f"Final decision: {terminal}."},
        ]

    def _medium_plan(self, risk: str, escalate: bool, available: List[str]) -> List[Dict]:
        terminal = "escalate_incident" if escalate else "dismiss_alert"
        plan = [
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Initial frame inspection."},
            {"action": "request_next_frame", "payload": None, "reasoning": "Navigating forward in timeline."},
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Inspecting second frame."},
            {"action": "request_next_frame", "payload": None, "reasoning": "Navigating to third frame."},
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Inspecting potential anomaly frame."},
            {"action": "classify_risk", "payload": risk, "reasoning": f"Temporal analysis complete. Risk: {risk}."},
            {"action": terminal, "payload": None, "reasoning": f"Final decision: {terminal}."},
        ]
        return plan

    def _hard_plan(self, risk: str, escalate: bool, available: List[str]) -> List[Dict]:
        terminal = "escalate_incident" if escalate else "dismiss_alert"

        # Build camera switching into the plan
        cams = self._camera_ids[1:3] if len(self._camera_ids) > 1 else []
        plan = [
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Inspecting initial camera."},
            {"action": "request_next_frame", "payload": None, "reasoning": "Checking temporal progression."},
            {"action": "inspect_current_frame", "payload": None, "reasoning": "Inspecting second frame."},
        ]
        # Switch to other cameras
        for cam in cams:
            plan.append({"action": "switch_camera", "payload": cam, "reasoning": f"Tracking suspect to {cam}."})
            plan.append({"action": "inspect_current_frame", "payload": None, "reasoning": f"Inspecting feed on {cam}."})
            plan.append({"action": "request_next_frame", "payload": None, "reasoning": "Checking temporal progression."})

        plan.extend([
            {"action": "classify_risk", "payload": risk, "reasoning": f"Cross-camera analysis complete. Risk: {risk}."},
            {"action": terminal, "payload": None, "reasoning": f"Final decision: {terminal}."},
        ])
        return plan


# ---------------------------------------------------------------------------
# Smart LLM Caller with Fallback
# ---------------------------------------------------------------------------

def call_llm_with_fallback(
    messages: List[Dict[str, Any]],
    model_chain: List[str],
) -> Tuple[Optional[str], str]:
    """
    Try each model in the fallback chain until one succeeds.
    Returns (response_text, model_used) or (None, "unavailable") if all fail.
    """
    last_error = ""

    for model_id in model_chain:
        for attempt in range(MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=200,
                )
                text = completion.choices[0].message.content or ""
                if text.strip():
                    return text, model_id
            except Exception as exc:
                err_str = str(exc)
                last_error = err_str

                # 402 = credits depleted → skip to next model immediately
                if "402" in err_str or "Payment Required" in err_str:
                    logger.warning("Model %s: credits depleted (402). Next model...", model_id)
                    break
                # 429 = rate limited
                if "429" in err_str or "Rate" in err_str.lower():
                    logger.warning("Model %s: rate limited (429). Next model...", model_id)
                    time.sleep(1)
                    break
                # 401 = auth error
                if "401" in err_str or "Unauthorized" in err_str:
                    logger.warning("Model %s: unauthorized (401). Next model...", model_id)
                    break
                # 404 = model not found
                if "404" in err_str or "Not Found" in err_str:
                    logger.warning("Model %s: not available (404). Next model...", model_id)
                    break
                # Other errors → retry with backoff
                logger.warning("Model %s attempt %d failed: %s", model_id, attempt + 1, err_str[:100])
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
        else:
            continue

    # All models exhausted
    logger.warning("All LLM models exhausted. Using deterministic baseline agent. Last error: %s", last_error[:100])
    return None, "unavailable"


# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------

def build_user_content(obs_data: Dict[str, Any], info: Dict[str, Any]) -> Any:
    """
    Build multimodal user content for VLM inference.
    Returns list of content parts (text + image) for vision models,
    or plain text string for text-only fallback.
    """
    obs = obs_data
    parts = [
        f"Camera: {obs['camera_id']}",
        f"Step: {obs['step']}",
        f"Alert Level: {obs.get('alert_level', 'unknown')}",
        f"Scene: {obs.get('context', 'No description available.')}",
        f"Actions: {', '.join(obs.get('available_actions', []))}",
    ]
    meta = obs.get("metadata", {})
    if meta.get("cameras_visited"):
        parts.append(f"Cameras visited: {', '.join(meta['cameras_visited'])}")
    if meta.get("zoom"):
        parts.append("Zoom: ACTIVE")

    if info:
        if "feedback" in info:
            parts.append(f"Feedback: {info['feedback']}")
        if "cumulative_reward" in info:
            parts.append(f"Score: {info['cumulative_reward']:.3f}")

    text_prompt = "\n".join(parts)

    # Build multimodal content with the frame image
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": text_prompt},
    ]
    frame_b64 = obs.get("frame_b64", "")
    if frame_b64:
        screenshot_uri = f"data:image/png;base64,{frame_b64}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": screenshot_uri},
        })
    return user_content


def parse_agent_response(text: str) -> Dict[str, Any]:
    """Parse the LLM's JSON response into an action dict. Handles formatting issues."""
    text = text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    brace_match = re.search(r"\{[^{}]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort — try to extract action from plain text
    for action in [
        "escalate_incident", "dismiss_alert", "classify_risk",
        "inspect_current_frame", "request_previous_frame", "request_next_frame",
        "switch_camera", "zoom_region",
    ]:
        if action in text.lower():
            return {"action": action, "payload": None, "reasoning": "Parsed from text."}

    return {"action": "inspect_current_frame", "payload": None, "reasoning": "Fallback parse."}


def extract_action_and_payload(parsed: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract (action_type, payload) from parsed response, handling colon-delimited actions."""
    action_raw = parsed.get("action", "inspect_current_frame")
    payload = parsed.get("payload")

    # Handle "switch_camera:cam-02" style
    if ":" in action_raw:
        parts = action_raw.split(":", 1)
        action_type = parts[0]
        if payload is None:
            payload = parts[1]
    else:
        action_type = action_raw

    return action_type, payload


def safe_env_step(env: EnvClient, action_type: str, payload: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Execute env.step() with error handling.
    Returns the step result or None if the action failed.
    """
    try:
        return env.step(action_type, payload=payload)
    except requests.exceptions.HTTPError as exc:
        logger.warning("env.step(%s, %s) failed: %s", action_type, payload, str(exc)[:100])
        return None
    except Exception as exc:
        logger.warning("env.step(%s, %s) unexpected error: %s", action_type, payload, str(exc)[:80])
        return None


def run_episode(
    env: EnvClient,
    task_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single full episode. Returns the grading result dict."""
    t0 = time.time()
    baseline_agent = DeterministicBaselineAgent()
    using_baseline = False

    # Reset environment
    reset_data = env.reset(task_id)
    obs_data = reset_data["observation"]
    info = reset_data.get("info", {})

    # Initialize baseline agent with task info
    baseline_agent.reset(info)

    # === MANDATORY STRUCTURED LOG: [START] ===
    start_log = {
        "task_id": info.get("task_id", task_id),
        "difficulty": info.get("difficulty", "unknown"),
        "title": info.get("title", "Unknown"),
        "max_steps": info.get("max_steps", 15),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[START] {json.dumps(start_log)}", flush=True)

    if verbose:
        logger.info("=" * 60)
        logger.info("EPISODE START: %s", info.get("title", "Unknown"))
        logger.info("Task: %s  Difficulty: %s", info.get("task_id"), info.get("difficulty"))
        logger.info("=" * 60)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    step = 0
    max_steps = info.get("max_steps", 15)
    terminated = False
    truncated = False
    active_model = MODEL_CHAIN[0]
    last_action_type = None

    while not terminated and not truncated and step < max_steps:
        # Check timeout
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > TIMEOUT_MINUTES:
            logger.warning("Timeout reached (%.1f min). Forcing escalation.", elapsed_min)
            safe_env_step(env, "escalate_incident")
            break

        action_type: str
        payload: Optional[str] = None

        if not using_baseline:
            # Build observation content for LLM
            user_content = build_user_content(obs_data, info)
            messages.append({"role": "user", "content": user_content})

            # Query LLM with fallback chain
            agent_text, active_model = call_llm_with_fallback(messages, MODEL_CHAIN)

            if agent_text is None:
                # LLM unavailable — switch to deterministic baseline for rest of episode
                using_baseline = True
                logger.info("Switching to deterministic baseline agent for this episode.")
                # Remove the last user message (LLM never saw it)
                messages.pop()
            else:
                # Replace multimodal content with text-only in history to save tokens
                if isinstance(user_content, list):
                    text_only = [p for p in user_content if p.get("type") == "text"]
                    messages[-1] = {"role": "user", "content": text_only[0]["text"] if text_only else ""}
                messages.append({"role": "assistant", "content": agent_text})

                parsed = parse_agent_response(agent_text)
                action_type, payload = extract_action_and_payload(parsed)

        if using_baseline:
            # Use the deterministic baseline agent
            baseline_action = baseline_agent.get_action(obs_data)
            action_type = baseline_action["action"]
            payload = baseline_action.get("payload")
            if ":" in action_type:
                parts = action_type.split(":", 1)
                action_type = parts[0]
                if payload is None:
                    payload = parts[1]
            active_model = "baseline"

        # Avoid repeating the exact same action (causes server 400)
        if action_type == last_action_type and action_type == "inspect_current_frame":
            # Rotate to a different action
            available = obs_data.get("available_actions", [])
            for alt in ["request_next_frame", "request_previous_frame", "classify_risk", "escalate_incident"]:
                if alt in available and alt != last_action_type:
                    action_type = alt
                    if alt == "classify_risk":
                        payload = "suspicious"
                    break

        # Rate-limiting for API calls
        if not using_baseline:
            time.sleep(2)

        if verbose:
            model_short = active_model.split("/")[-1][:25] if "/" in active_model else active_model
            logger.info(
                "Step %d │ %-28s │ Payload: %-12s │ Model: %s",
                step, action_type, payload or "None", model_short,
            )

        # Execute action with error recovery
        step_result = safe_env_step(env, action_type, payload=payload)

        if step_result is None:
            # Action failed — try a safe fallback action
            available = obs_data.get("available_actions", [])
            fallback_tried = False
            for fallback_action in ["request_next_frame", "classify_risk", "escalate_incident", "dismiss_alert"]:
                if fallback_action in available and fallback_action != action_type:
                    fb_payload = "suspicious" if fallback_action == "classify_risk" else None
                    step_result = safe_env_step(env, fallback_action, fb_payload)
                    if step_result is not None:
                        logger.info("  Recovered with fallback action: %s", fallback_action)
                        action_type = fallback_action
                        fallback_tried = True
                        break
            if step_result is None:
                logger.error("  All fallback actions failed. Skipping step.")
                step += 1
                continue

        obs_data = step_result["observation"]
        reward = step_result["reward"]
        terminated = step_result["terminated"]
        truncated = step_result["truncated"]
        info = step_result.get("info", {})
        last_action_type = action_type

        # === MANDATORY STRUCTURED LOG: [STEP] ===
        step_log = {
            "step": step,
            "action": action_type,
            "payload": payload,
            "reward": round(reward, 4),
            "cumulative_reward": round(info.get("cumulative_reward", 0.0), 4),
            "terminated": terminated,
            "truncated": truncated,
            "model": active_model.split("/")[-1] if "/" in active_model else active_model,
        }
        print(f"[STEP] {json.dumps(step_log)}", flush=True)

        if verbose:
            logger.info(
                "         │ Reward: %+.3f │ Feedback: %s",
                reward, info.get("feedback", "")[:80],
            )

        step += 1

    # Grade
    elapsed = time.time() - t0
    try:
        grade_result = env.grade()
    except Exception as exc:
        logger.error("Grading failed: %s", exc)
        grade_result = {"score": 0.0, "error": str(exc)}

    grade_result["elapsed_seconds"] = round(elapsed, 2)
    grade_result["total_steps"] = step
    grade_result["model_used"] = active_model

    # === MANDATORY STRUCTURED LOG: [END] ===
    end_log = {
        "task_id": grade_result.get("task_id", task_id),
        "difficulty": grade_result.get("difficulty", "unknown"),
        "score": grade_result.get("score", 0.0),
        "steps": step,
        "elapsed_seconds": round(elapsed, 2),
        "model": active_model.split("/")[-1] if "/" in active_model else active_model,
        "breakdown": grade_result.get("breakdown", {}),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[END] {json.dumps(end_log)}", flush=True)

    if verbose:
        logger.info("-" * 60)
        logger.info("EPISODE COMPLETE")
        logger.info("Score:    %.4f", grade_result.get("score", 0))
        logger.info("Steps:    %d", step)
        logger.info("Time:     %.1fs", elapsed)
        logger.info("Model:    %s", active_model)
        if "breakdown" in grade_result:
            for k, v in grade_result["breakdown"].items():
                logger.info("  %-30s %.3f", k, v)
        logger.info("-" * 60)

    return grade_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run inference across all available tasks and report aggregate scores."""
    logger.info("SentinelOps Inference — Starting")
    logger.info("API_BASE_URL: %s", API_BASE_URL)
    logger.info("MODEL_NAME:   %s", MODEL_NAME)
    logger.info("ENV_URL:      %s", ENV_URL)
    logger.info("Model chain:  %s", " → ".join(m.split('/')[-1][:30] for m in MODEL_CHAIN))

    env = EnvClient(ENV_URL)

    # Discover tasks
    try:
        tasks = env.list_tasks()
    except Exception as exc:
        logger.error("Cannot connect to environment: %s", exc)
        logger.info("Ensure the SentinelOps server is running at %s", ENV_URL)
        sys.exit(1)

    logger.info("Discovered %d tasks", len(tasks))

    results: List[Dict[str, Any]] = []
    total_score = 0.0

    for task_meta in tasks:
        tid = task_meta["task_id"]
        logger.info("\n▶ Running task: %s (%s)", tid, task_meta.get("difficulty", "?"))
        result = run_episode(env, task_id=tid, verbose=True)
        results.append(result)
        total_score += result.get("score", 0)

    # Aggregate
    n = len(results) or 1
    avg_score = total_score / n

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    logger.info("Tasks run:     %d", len(results))
    logger.info("Total score:   %.4f", total_score)
    logger.info("Average score: %.4f", avg_score)
    logger.info("=" * 60)

    # Print reproducible JSON scores
    print("\n--- REPRODUCIBLE SCORES ---")
    output = {
        "total_tasks": len(results),
        "total_score": round(total_score, 4),
        "average_score": round(avg_score, 4),
        "per_task": [
            {
                "task_id": r.get("task_id", "unknown"),
                "difficulty": r.get("difficulty", "unknown"),
                "score": r.get("score", 0),
                "steps": r.get("total_steps", 0),
                "elapsed_seconds": r.get("elapsed_seconds", 0),
                "model_used": r.get("model_used", "unknown"),
            }
            for r in results
        ],
    }
    print(json.dumps(output, indent=2))
    print("--- END SCORES ---")

    return output


if __name__ == "__main__":
    main()
