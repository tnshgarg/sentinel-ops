"""
SentinelOps — Configuration Module.

Centralises all tuneable parameters, file paths, and environment variables
so that every other module imports a single *frozen* config object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = _PROJECT_ROOT / "assets"
FRAMES_DIR = ASSETS_DIR / "frames"
SEQUENCES_DIR = ASSETS_DIR / "sequences"
TASKS_DIR = _PROJECT_ROOT / "tasks"


# ---------------------------------------------------------------------------
# Reward Table
# ---------------------------------------------------------------------------

DEFAULT_REWARD_TABLE: Dict[str, float] = {
    "correct_anomaly_detection": 0.20,
    "correct_temporal_reasoning": 0.20,
    "correct_escalation": 0.30,
    "fast_response": 0.10,
    "false_positive": -0.20,
    "missed_anomaly": -0.40,
    "random_action_spam": -0.10,
    "unsafe_dismissal": -0.30,
    "correct_camera_switch": 0.20,
    "correct_risk_classification": 0.30,
    "unnecessary_zoom": -0.05,
    "redundant_inspect": -0.05,
}


# ---------------------------------------------------------------------------
# Settings (populated from env vars + defaults)
# ---------------------------------------------------------------------------

class SentinelOpsSettings(BaseSettings):
    """
    Application-wide settings.

    Values are read from environment variables (upper-case) and fall back
    to the defaults declared here.
    """

    # --- OpenEnv / HF deployment ---
    api_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        description="Base URL for the OpenAI-compatible inference endpoint.",
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        description="Model identifier used for agentic inference.",
    )
    hf_token: str = Field(
        default="",
        description="Hugging Face API token.",
    )

    # --- Environment tuning ---
    max_episode_steps: int = Field(default=15, ge=1)
    default_camera: str = "cam-01"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    inference_timeout_minutes: int = Field(default=20, ge=1)

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = Field(default=7860, ge=1, le=65535)
    env_url: str = Field(
        default="http://localhost:7860",
        description="URL of the SentinelOps environment server (used by inference).",
    )

    # --- Paths (derived) ---
    project_root: Path = _PROJECT_ROOT
    assets_dir: Path = ASSETS_DIR
    frames_dir: Path = FRAMES_DIR
    sequences_dir: Path = SEQUENCES_DIR
    tasks_dir: Path = TASKS_DIR

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",),
    }


# Singleton — importable as `from config import settings`
settings = SentinelOpsSettings()
