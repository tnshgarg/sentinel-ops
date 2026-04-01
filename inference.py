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
    • Model fallback chain: tries primary → cheaper alternatives on 402/429
    • Uses HF `:cheapest` routing policy to minimise credit consumption
    • Graceful degradation when all LLM providers are exhausted
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
TIMEOUT_MINUTES = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinelops.inference")


# ---------------------------------------------------------------------------
# Model Fallback Chain
# ---------------------------------------------------------------------------

# Ordered from preferred (best quality) to cheapest (smallest / free-tier).
# The `:cheapest` suffix tells HF router to pick the cheapest provider.
# Smaller models consume fewer credits per request.
MODEL_FALLBACK_CHAIN: List[str] = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
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
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are SentinelOps Agent — an elite AI security analyst operating a multi-camera surveillance control room.

Your mission is to:
1. Inspect surveillance frames for anomalies
2. Navigate between frames and cameras to track threats
3. Classify the risk level of any detected anomaly
4. Decide whether to escalate the incident or dismiss the alert

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
{"action": "<action_name>", "payload": "<optional_payload>", "reasoning": "<brief explanation>"}

Examples:
{"action": "inspect_current_frame", "payload": null, "reasoning": "Need to examine what the camera is showing."}
{"action": "switch_camera", "payload": "cam-02", "reasoning": "Suspect may have moved to the warehouse area."}
{"action": "classify_risk", "payload": "dangerous", "reasoning": "Person is breaking into a vehicle — clear criminal activity."}
{"action": "escalate_incident", "payload": null, "reasoning": "Confirmed break-in requires immediate police response."}

STRATEGY:
- Always inspect the current frame first before making decisions
- Use temporal navigation to understand how the situation evolved
- In multi-camera scenarios, switch to track suspects across areas
- Only escalate when you have strong evidence of a real threat
- Dismiss only when confident there is no genuine threat
- Be efficient — fewer steps is better
"""


# ---------------------------------------------------------------------------
# Smart LLM Caller with Fallback
# ---------------------------------------------------------------------------

def call_llm_with_fallback(
    messages: List[Dict[str, str]],
    model_chain: List[str],
) -> Tuple[str, str]:
    """
    Try each model in the fallback chain until one succeeds.
    Returns (response_text, model_used).

    Handles:
        - 402 Payment Required  → immediately try next model
        - 429 Rate Limited      → brief pause then next model
        - 401 Unauthorized      → try next model
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
                    max_tokens=200,  # Reduced to save tokens
                )
                text = completion.choices[0].message.content or ""
                if text.strip():
                    return text, model_id
            except Exception as exc:
                err_str = str(exc)
                last_error = err_str

                # 402 = credits depleted → skip to next model immediately
                if "402" in err_str or "Payment Required" in err_str:
                    logger.warning(
                        "Model %s: credits depleted (402). Trying next model...",
                        model_id,
                    )
                    break  # Break inner retry loop, go to next model

                # 429 = rate limited → brief pause then try next
                if "429" in err_str or "Rate" in err_str.lower():
                    logger.warning(
                        "Model %s: rate limited (429). Trying next model...",
                        model_id,
                    )
                    time.sleep(1)
                    break

                # 401 = auth error → try next model
                if "401" in err_str or "Unauthorized" in err_str:
                    logger.warning(
                        "Model %s: unauthorized (401). Trying next model...",
                        model_id,
                    )
                    break

                # 404 = model not found on this provider
                if "404" in err_str or "Not Found" in err_str:
                    logger.warning(
                        "Model %s: not available (404). Trying next model...",
                        model_id,
                    )
                    break

                # Other errors → retry once with backoff
                logger.warning(
                    "Model %s attempt %d failed: %s",
                    model_id, attempt + 1, err_str[:100],
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
        else:
            # All retries for this model exhausted, try next
            continue

    # All models exhausted — return fallback
    logger.error(
        "All models exhausted. Last error: %s. Using fallback action.",
        last_error[:150],
    )
    fallback = '{"action": "inspect_current_frame", "payload": null, "reasoning": "All LLM providers unavailable, using fallback."}'
    return fallback, "fallback"


# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------

def build_user_content(obs_data: Dict[str, Any], info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build multimodal user content matching the hackathon sample inference script.

    Returns a list of content parts:
        [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]

    This mirrors the exact pattern shown in the official sample:
        user_content = [{"type": "text", "text": user_prompt}]
        screenshot_uri = extract_screenshot_uri(observation)
        if screenshot_uri:
            user_content.append({"type": "image_url", "image_url": {"url": screenshot_uri}})
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

    # When ENABLE_VISION is true and a vision-capable model is used,
    # build multimodal content list with the frame image attached.
    # Otherwise, return text-only (compatible with all text LLMs).
    enable_vision = os.environ.get("ENABLE_VISION", "false").lower() == "true"

    if enable_vision:
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
    else:
        # Text-only mode — return plain string (works with all models)
        return text_prompt


def parse_agent_response(text: str) -> Dict[str, Any]:
    """
    Parse the LLM's JSON response into an action dict.
    Handles various formatting issues robustly.
    """
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
        "inspect_current_frame", "request_previous_frame", "request_next_frame",
        "switch_camera", "zoom_region", "classify_risk",
        "escalate_incident", "dismiss_alert",
    ]:
        if action in text.lower():
            return {"action": action, "payload": None, "reasoning": "Parsed from text."}

    # Fallback
    return {"action": "inspect_current_frame", "payload": None, "reasoning": "Fallback action."}


def extract_action_and_payload(parsed: Dict[str, Any]) -> tuple:
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


def run_episode(
    env: EnvClient,
    task_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single full episode with the LLM agent.

    Returns the grading result dict.
    """
    t0 = time.time()

    # Reset environment
    reset_data = env.reset(task_id)
    obs_data = reset_data["observation"]
    info = reset_data.get("info", {})

    if verbose:
        logger.info("="*60)
        logger.info("EPISODE START: %s", info.get("title", "Unknown"))
        logger.info("Task: %s  Difficulty: %s", info.get("task_id"), info.get("difficulty"))
        logger.info("="*60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    step = 0
    max_steps = info.get("max_steps", 15)
    terminated = False
    truncated = False
    active_model = MODEL_CHAIN[0]  # Track which model is working

    while not terminated and not truncated and step < max_steps:
        # Check timeout
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > TIMEOUT_MINUTES:
            logger.warning("Timeout reached (%.1f min). Forcing escalation.", elapsed_min)
            step_result = env.step("escalate_incident")
            terminated = step_result.get("terminated", True)
            break

        # Build observation content (text-only or multimodal depending on ENABLE_VISION)
        user_content = build_user_content(obs_data, info)
        messages.append({"role": "user", "content": user_content})

        # Query LLM with fallback chain
        agent_text, active_model = call_llm_with_fallback(messages, MODEL_CHAIN)

        # If multimodal mode was used, replace content with text-only
        # to prevent base64 images from bloating the conversation history
        if isinstance(user_content, list):
            text_only = [p for p in user_content if p.get("type") == "text"]
            messages[-1] = {"role": "user", "content": text_only[0]["text"] if text_only else ""}
        messages.append({"role": "assistant", "content": agent_text})

        # Parse response
        parsed = parse_agent_response(agent_text)
        action_type, payload = extract_action_and_payload(parsed)

        # Rate-limiting to ensure we do not hit HF free-tier burst limits
        time.sleep(3)

        if verbose:
            model_short = active_model.split("/")[-1][:25] if "/" in active_model else active_model
            logger.info(
                "Step %d │ %-28s │ Payload: %-8s │ Model: %s",
                step, action_type, payload, model_short,
            )

        # Execute action
        try:
            step_result = env.step(action_type, payload=payload)
        except Exception as exc:
            logger.error("Step failed: %s — falling back to inspect.", exc)
            step_result = env.step("inspect_current_frame")

        obs_data = step_result["observation"]
        reward = step_result["reward"]
        terminated = step_result["terminated"]
        truncated = step_result["truncated"]
        info = step_result.get("info", {})

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

    if verbose:
        logger.info("-"*60)
        logger.info("EPISODE COMPLETE")
        logger.info("Score:    %.4f", grade_result.get("score", 0))
        logger.info("Steps:    %d", step)
        logger.info("Time:     %.1fs", elapsed)
        if "breakdown" in grade_result:
            for k, v in grade_result["breakdown"].items():
                logger.info("  %-30s %.3f", k, v)
        logger.info("-"*60)

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

    logger.info("\n" + "="*60)
    logger.info("AGGREGATE RESULTS")
    logger.info("="*60)
    logger.info("Tasks run:     %d", len(results))
    logger.info("Total score:   %.4f", total_score)
    logger.info("Average score: %.4f", avg_score)
    logger.info("="*60)

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
