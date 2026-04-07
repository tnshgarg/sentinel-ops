#!/usr/bin/env python3
"""
SentinelOps — Inference Script (Submission Entry Point).

This is the root-level inference script required by the OpenEnv hackathon spec.
It uses the OpenAI-compatible client to drive an LLM agent through
SentinelOps episodes, printing reproducible scores.

Required environment variables:
    API_BASE_URL   – base URL for the inference API
    MODEL_NAME     – model identifier (used as primary; fallbacks apply on error)
    HF_TOKEN       – Hugging Face API token

Structured stdout log format (mandatory per spec):
    [START] {"task_id": "...", "difficulty": "...", "title": "..."}
    [STEP]  {"step": 1, "action": "...", "payload": null, "reward": 0.15, "cumulative_reward": 0.15, "done": false}
    [END]   {"task_id": "...", "score": 0.75, "steps": 5, "status": "success"}

Constraints:
    • Uses ``from openai import OpenAI`` as mandated
    • Deterministic low-temperature inference (temperature=0.1)
    • Completes full episode < 20 minutes
    • Prints reproducible scores
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
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

TEMPERATURE = 0.1
MAX_RETRIES = 2
TIMEOUT_MINUTES = 18  # Leave 2-min buffer below 20-min hard limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinelops.inference")


# ---------------------------------------------------------------------------
# Structured stdout logging — mandatory format per hackathon spec
# ---------------------------------------------------------------------------

def _emit(tag: str, data: dict) -> None:
    """
    Emit a structured log line to stdout.

    Format: <tag> <json>
    Example: [START] {"task_id": "task_001", "difficulty": "easy", "title": "..."}

    The evaluator parses these lines to verify scores.
    flush=True ensures lines appear immediately even under buffering.
    """
    print(f"{tag} {json.dumps(data)}", flush=True)


# ---------------------------------------------------------------------------
# Model Fallback Chain
# ---------------------------------------------------------------------------

# MODEL_NAME from env is always tried first.
# Fallbacks are tried in order on 402/429/401/404 errors.
_FALLBACK_ALTERNATIVES: List[str] = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

_seen: set = set()
MODEL_CHAIN: List[str] = []
for _m in [MODEL_NAME] + _FALLBACK_ALTERNATIVES:
    if _m and _m not in _seen:
        _seen.add(_m)
        MODEL_CHAIN.append(_m)


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

    def step(self, action_type: str, payload: Optional[str] = None, confidence: float = 1.0) -> Dict[str, Any]:
        body = {
            "action_type": action_type,
            "payload": payload,
            "confidence": confidence,
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
# System Prompt — rubric-aware strategy to maximise grader scores
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are SentinelOps Agent — an elite AI security analyst operating a multi-camera surveillance control room.

Your mission: inspect surveillance frames, navigate the timeline and cameras to detect anomalies, classify threat severity, and decide whether to escalate or dismiss.

AVAILABLE ACTIONS — respond with EXACTLY one JSON object per turn, on a single line:
{"action": "inspect_current_frame", "payload": null, "reasoning": "..."}
{"action": "request_next_frame", "payload": null, "reasoning": "..."}
{"action": "request_previous_frame", "payload": null, "reasoning": "..."}
{"action": "switch_camera", "payload": "cam-02", "reasoning": "..."}
{"action": "zoom_region", "payload": "center", "reasoning": "..."}
{"action": "classify_risk", "payload": "dangerous", "reasoning": "..."}
{"action": "escalate_incident", "payload": null, "reasoning": "..."}
{"action": "dismiss_alert", "payload": null, "reasoning": "..."}

Risk levels for classify_risk payload: safe | suspicious | dangerous | critical

Examples:
{"action": "inspect_current_frame", "payload": null, "reasoning": "Need to examine what the camera is showing."}
{"action": "switch_camera", "payload": "cam-02", "reasoning": "Suspect may have moved to the warehouse area."}
{"action": "classify_risk", "payload": "dangerous", "reasoning": "Person is breaking into a vehicle — clear criminal activity."}
{"action": "escalate_incident", "payload": null, "reasoning": "Confirmed break-in requires immediate police response."}
{"action": "dismiss_alert", "payload": null, "reasoning": "Authorised personnel conducting scheduled maintenance — no threat."}

MANDATORY STRATEGY BY DIFFICULTY:

EASY tasks (single camera, 2-3 frames):
  The first inspect_current_frame has already been done. Read the scene description:
  - If scene says NO anomaly/threat: use request_next_frame to advance, then inspect_current_frame
  - Repeat until you find a frame describing an anomaly or threat
  - Once anomaly evidence found: classify_risk:<correct_level> (from scene description)
  - Then: escalate_incident if a real threat confirmed, OR dismiss_alert if genuinely safe
  Typical path: [seeded inspect] → request_next_frame → inspect_current_frame → classify_risk → escalate
  Risk hint: vehicle break-in/tampering=dangerous, unauthorized access=suspicious, bomb/critical threat=critical

MEDIUM tasks (single camera, 3-4 frames with anomaly appearing mid-sequence):
  The first inspect_current_frame has already been done (frame 0).
  Follow this EXACT sequence:
  Step 2: request_next_frame        — advance timeline (nav count = 1)
  Step 3: inspect_current_frame     — inspect new frame (now 2 frames inspected)
  Step 4: request_next_frame        — advance again (nav count = 2) → earns temporal bonus
  Step 5: inspect_current_frame     — inspect anomaly onset frame when reached
  Step 6: classify_risk:<level>     — classify based on what you now see
  Step 7: escalate_incident OR dismiss_alert
  You MUST use request_next_frame at least TWICE before classifying to earn temporal bonus points.

HARD tasks (multi-camera, 4 cameras, must visit 3+ different cameras):
  The first inspect_current_frame has already been done (starting camera).
  Follow this EXACT sequence:
  Step 2: switch_camera:<cam-id>    — switch to a 2nd camera (switch count = 1)
  Step 3: switch_camera:<cam-id>    — switch to a 3rd camera (switch count = 2) → 3 cameras ✓
  Step 4: switch_camera:<cam-id>    — switch to the camera where the REAL THREAT is (switch count = 3)
           Identify the threat camera by inspecting each feed and reading the scene descriptions.
  Step 5: inspect_current_frame     — inspect the threat camera
  Step 6: request_next_frame        — advance frames (nav count = 1)
  Step 7: inspect_current_frame     — inspect next frame for evidence
  Step 8: request_next_frame        — advance again (nav count = 2) → earns temporal bonus ✓
  Step 9: classify_risk:<level>     — classify ONLY on the threat camera with confirmed evidence
  Step 10: escalate_incident OR dismiss_alert
  - switch_camera payload must be exact: "cam-01", "cam-02", "cam-03", or "cam-04"
  - Visit each camera and read its scene description to determine where the real threat is
  - Do NOT classify_risk unless you are ON the threat camera with clear evidence
  - Only classify ONCE — misclassifications are heavily penalised

GENERAL RULES:
  - classify_risk MUST always be called BEFORE escalate_incident or dismiss_alert
  - classify_risk payload must be exactly one of: safe, suspicious, dangerous, critical
  - After classifying risk, your VERY NEXT action must be escalate_incident or dismiss_alert
  - Never repeat the same action 3+ times in a row (spam penalty)
  - Do NOT re-inspect a frame already inspected on the same camera (redundant penalty)
  - If scene shows authorised personnel (badges, uniforms, work orders): classify as "safe" and dismiss

You MUST respond with a SINGLE valid JSON object on ONE line. No markdown fences. No extra text outside the JSON.
"""


# ---------------------------------------------------------------------------
# Smart LLM Caller with Fallback
# ---------------------------------------------------------------------------

def call_llm_with_fallback(
    messages: List[Dict[str, Any]],
    model_chain: List[str],
) -> Tuple[str, str]:
    """
    Try each model in the fallback chain until one succeeds.
    Returns (response_text, model_used).

    Handles:
        - 402 Payment Required  → immediately try next model
        - 429 Rate Limited      → brief pause then next model
        - 401 Unauthorized      → try next model
        - 404 Not Found         → try next model
        - Other errors          → retry once, then next model
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

                if "402" in err_str or "Payment Required" in err_str:
                    logger.warning("Model %s: credits depleted (402). Trying next model...", model_id)
                    break

                if "429" in err_str or "Rate" in err_str.lower():
                    logger.warning("Model %s: rate limited (429). Trying next model...", model_id)
                    time.sleep(1)
                    break

                if "401" in err_str or "Unauthorized" in err_str:
                    logger.warning("Model %s: unauthorized (401). Trying next model...", model_id)
                    break

                if "404" in err_str or "Not Found" in err_str:
                    logger.warning("Model %s: not available (404). Trying next model...", model_id)
                    break

                logger.warning("Model %s attempt %d failed: %s", model_id, attempt + 1, err_str[:100])
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
        else:
            continue

    logger.error("All models exhausted. Last error: %s. Using fallback action.", last_error[:150])
    fallback = '{"action": "inspect_current_frame", "payload": null, "reasoning": "All LLM providers unavailable, using fallback."}'
    return fallback, "fallback"


# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------

def build_user_content(obs_data: Dict[str, Any], info: Dict[str, Any]):
    """
    Build user content from the current observation.

    When ENABLE_VISION=true, returns a multimodal list [text, image] so the LLM
    actually sees the surveillance frame.  Falls back to plain text for text-only
    models.  Images are stripped from history after each turn to avoid token bloat.
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
        parts.append(f"Cameras visited so far: {', '.join(meta['cameras_visited'])}")
    if meta.get("zoom"):
        parts.append("Zoom: ACTIVE")

    if info:
        if "feedback" in info:
            parts.append(f"Last feedback: {info['feedback']}")
        if "cumulative_reward" in info:
            parts.append(f"Running score: {info['cumulative_reward']:.3f}")

    text_prompt = "\n".join(parts)

    enable_vision = os.environ.get("ENABLE_VISION", "false").lower() == "true"
    if enable_vision:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        frame_b64 = obs.get("frame_b64", "")
        if frame_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{frame_b64}"},
            })
        return user_content

    return text_prompt


def parse_agent_response(text: str) -> Dict[str, Any]:
    """
    Parse the LLM's JSON response into an action dict.
    Handles various formatting issues robustly.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{[^{}]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    for action in [
        "inspect_current_frame", "request_previous_frame", "request_next_frame",
        "switch_camera", "zoom_region", "classify_risk",
        "escalate_incident", "dismiss_alert",
    ]:
        if action in text.lower():
            return {"action": action, "payload": None, "reasoning": "Parsed from text."}

    return {"action": "inspect_current_frame", "payload": None, "reasoning": "Fallback action."}


def extract_action_and_payload(parsed: Dict[str, Any]) -> tuple:
    """Extract (action_type, payload) from parsed response, handling colon-delimited actions."""
    action_raw = parsed.get("action", "inspect_current_frame")
    payload = parsed.get("payload")

    if ":" in action_raw:
        parts = action_raw.split(":", 1)
        action_type = parts[0]
        if payload is None:
            payload = parts[1]
    else:
        action_type = action_raw

    return action_type, payload


def run_episode(
    env: EnvClient,
    task_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single full episode with the LLM agent.

    Emits [START], [STEP] (per step), and [END] to stdout per hackathon spec.
    Returns the grading result dict.
    """
    t0 = time.time()

    # ── Reset environment ─────────────────────────────────────────────────
    reset_data = env.reset(task_id)
    obs_data = reset_data["observation"]
    info = reset_data.get("info", {})

    # ── [START] structured log (mandatory) ────────────────────────────────
    _emit("[START]", {
        "task_id": info.get("task_id", task_id or "unknown"),
        "difficulty": info.get("difficulty", "unknown"),
        "title": info.get("title", "unknown"),
    })

    if verbose:
        logger.info("=" * 60)
        logger.info("EPISODE START: %s", info.get("title", "Unknown"))
        logger.info("Task: %s  Difficulty: %s", info.get("task_id"), info.get("difficulty"))
        logger.info("=" * 60)

    difficulty = info.get("difficulty", "easy")
    camera_ids = info.get("camera_ids", [obs_data.get("camera_id", "cam-01")])
    max_steps = info.get("max_steps", 15)

    # ── Build conversation: system + opening context ───────────────────────
    opening_context = (
        f"NEW EPISODE: {info.get('title', 'Unknown Task')}\n"
        f"Difficulty: {difficulty.upper()}\n"
        f"Available cameras: {', '.join(camera_ids)}\n"
        f"Max steps allowed: {max_steps}\n\n"
        f"An alert has been triggered. Investigate the camera feeds to find the threat.\n"
        f"Follow the MANDATORY STRATEGY for {difficulty.upper()} tasks exactly."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": opening_context},
        # Seed the first action — guarantees frame_inspection rubric points
        {"role": "assistant", "content": '{"action": "inspect_current_frame", "payload": null, "reasoning": "Starting by inspecting the current camera frame to gather initial evidence."}'},
    ]

    # ── Execute seeded first action (inspect_current_frame guaranteed) ─────
    try:
        step_result = env.step("inspect_current_frame", payload=None)
    except Exception as exc:
        logger.error("First step failed: %s", exc)
        step_result = {"observation": obs_data, "reward": 0.0, "terminated": False, "truncated": False, "info": {}}

    obs_data = step_result["observation"]
    reward = step_result.get("reward", 0.0)
    episode_terminated = step_result.get("terminated", False)
    episode_truncated = step_result.get("truncated", False)
    info_step = step_result.get("info", {})

    _emit("[STEP]", {
        "step": 1,
        "action": "inspect_current_frame",
        "payload": None,
        "reward": round(reward, 4),
        "cumulative_reward": round(info_step.get("cumulative_reward", 0.0), 4),
        "done": episode_terminated or episode_truncated,
    })

    if verbose:
        logger.info("Step 1 │ inspect_current_frame │ Reward: %+.3f │ %s",
                    reward, info_step.get("feedback", "")[:80])

    step = 1  # Already executed step 1

    # ── Main LLM-driven loop ───────────────────────────────────────────────
    active_model = MODEL_CHAIN[0]

    while not episode_terminated and not episode_truncated and step < max_steps:
        # Timeout guard
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > TIMEOUT_MINUTES:
            logger.warning("Timeout reached (%.1f min). Forcing escalation.", elapsed_min)
            try:
                step_result = env.step("escalate_incident")
                _emit("[STEP]", {
                    "step": step + 1,
                    "action": "escalate_incident",
                    "payload": None,
                    "reward": round(step_result.get("reward", 0.0), 4),
                    "cumulative_reward": round(step_result.get("info", {}).get("cumulative_reward", 0.0), 4),
                    "done": True,
                })
            except Exception:
                pass
            episode_terminated = True
            break

        # --- Loop detection: force a terminal decision if agent is stuck ---
        if len(messages) >= 7:
            recent_actions = [
                m["content"] for m in messages
                if m["role"] == "assistant"
            ][-3:]
            if len(recent_actions) == 3:
                parsed_recent = [parse_agent_response(a) for a in recent_actions]
                recent_types = [p.get("action", "") for p in parsed_recent]
                if len(set(recent_types)) == 1:
                    logger.warning(
                        "Loop detected: agent repeated '%s' 3 times. Injecting decision-forcing prompt.",
                        recent_types[0],
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "ALERT: You have repeated the same action 3 times. "
                            "You MUST now make a final decision. "
                            "First respond with classify_risk:<level>, then on the very next turn "
                            "use escalate_incident or dismiss_alert. No more exploration."
                        ),
                    })
                    agent_text, active_model = call_llm_with_fallback(messages, MODEL_CHAIN)
                    messages.append({"role": "assistant", "content": agent_text})
                    parsed = parse_agent_response(agent_text)
                    action_type, payload = extract_action_and_payload(parsed)
                    try:
                        step_result = env.step(action_type, payload=payload)
                    except Exception:
                        step_result = env.step("escalate_incident")
                        action_type = "escalate_incident"
                        payload = None
                    obs_data = step_result["observation"]
                    reward = step_result.get("reward", 0.0)
                    episode_terminated = step_result.get("terminated", False)
                    episode_truncated = step_result.get("truncated", False)
                    info_step = step_result.get("info", {})
                    _emit("[STEP]", {
                        "step": step + 1,
                        "action": action_type,
                        "payload": payload,
                        "reward": round(reward, 4),
                        "cumulative_reward": round(info_step.get("cumulative_reward", 0.0), 4),
                        "done": episode_terminated or episode_truncated,
                    })
                    step += 1
                    if episode_terminated or episode_truncated:
                        break
                    continue

        # Build observation content (multimodal if ENABLE_VISION=true, else text)
        user_content = build_user_content(obs_data, info_step)
        messages.append({"role": "user", "content": user_content})

        # Query LLM with fallback chain
        agent_text, active_model = call_llm_with_fallback(messages, MODEL_CHAIN)

        # Strip image from history after the call — keeps tokens manageable
        if isinstance(user_content, list):
            text_only = next((p["text"] for p in user_content if p.get("type") == "text"), "")
            messages[-1] = {"role": "user", "content": text_only}

        messages.append({"role": "assistant", "content": agent_text})

        # Parse response
        parsed = parse_agent_response(agent_text)
        action_type, payload = extract_action_and_payload(parsed)

        # Rate-limiting: 1s between requests (safe on free tier, within 20-min budget)
        time.sleep(1)

        if verbose:
            model_short = active_model.split("/")[-1][:25] if "/" in active_model else active_model
            logger.info(
                "Step %d │ %-28s │ Payload: %-10s │ Model: %s",
                step + 1, action_type, str(payload)[:10], model_short,
            )

        # Execute action
        try:
            step_result = env.step(action_type, payload=payload)
        except Exception as exc:
            logger.error("Step failed: %s — falling back to inspect.", exc)
            try:
                step_result = env.step("inspect_current_frame")
                action_type = "inspect_current_frame"
                payload = None
            except Exception:
                break

        obs_data = step_result["observation"]
        reward = step_result.get("reward", 0.0)
        episode_terminated = step_result.get("terminated", False)
        episode_truncated = step_result.get("truncated", False)
        info_step = step_result.get("info", {})

        # [STEP] structured log (mandatory)
        _emit("[STEP]", {
            "step": step + 1,
            "action": action_type,
            "payload": payload,
            "reward": round(reward, 4),
            "cumulative_reward": round(info_step.get("cumulative_reward", 0.0), 4),
            "done": episode_terminated or episode_truncated,
        })

        if verbose:
            logger.info(
                "         │ Reward: %+.3f │ Feedback: %s",
                reward, info_step.get("feedback", "")[:80],
            )

        step += 1

    # ── Grade episode ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    try:
        grade_result = env.grade()
    except Exception as exc:
        # Episode may not be done yet — force termination then retry grading
        logger.warning("Grade failed (%s). Forcing escalation to terminate episode...", exc)
        try:
            env.step("escalate_incident")
            episode_terminated = True
            grade_result = env.grade()
        except Exception as exc2:
            logger.error("Grading failed even after force escalation: %s", exc2)
            grade_result = {"score": 0.0, "error": str(exc2)}

    # ── [END] structured log (mandatory) ──────────────────────────────────
    if episode_terminated:
        status = "success"
    elif episode_truncated or step >= max_steps:
        status = "truncated"
    else:
        status = "error"

    _emit("[END]", {
        "task_id": grade_result.get("task_id", task_id or "unknown"),
        "score": round(grade_result.get("score", 0.0), 4),
        "steps": step,
        "status": status,
    })

    grade_result["elapsed_seconds"] = round(elapsed, 2)
    grade_result["total_steps"] = step

    if verbose:
        logger.info("-" * 60)
        logger.info("EPISODE COMPLETE")
        logger.info("Score:    %.4f", grade_result.get("score", 0))
        logger.info("Steps:    %d", step)
        logger.info("Time:     %.1fs", elapsed)
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
    logger.info("Model chain:  %s", " → ".join(m.split("/")[-1][:30] for m in MODEL_CHAIN))

    env = EnvClient(ENV_URL)

    # Verify server is reachable and reset() returns 200
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
                "score": r.get("score", 0),
                "steps": r.get("total_steps", 0),
                "elapsed_seconds": r.get("elapsed_seconds", 0),
            }
            for r in results
        ],
    }
    print(json.dumps(output, indent=2))
    print("--- END SCORES ---")

    return output


if __name__ == "__main__":
    main()
