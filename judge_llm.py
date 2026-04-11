#!/usr/bin/env python3
"""
SentinelOps — LLM-as-a-Judge Evaluation Script.

This script implements qualitative evaluation of agent trajectories using
frontier VLMs (e.g., Llama 3.2 Vision). It evaluates reasoning, strategy,
and professionalism, providing a 'Soft Score' to supplement the deterministic 
OpenEnv rubric.

Usage:
    python judge_llm.py --state-file path/to/episode.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

JUDGE_PROMPT = """You are 'Sentinel-Judge', an elite surveillance operations supervisor.
Your task is to evaluate the performance of an AI Security Agent based on its action trajectory.

AGENT TASK: Monitor CCTV feeds, detect anomalies, classify risk (safe, suspicious, dangerous, critical), and decide to escalate or dismiss.

EVALUATION CRITERIA:
1. Reasoning Depth: Did the agent's internal reasoning show deep understanding of the video context?
2. Strategic Navigation: Did it switch cameras and navigate timelines logically to track the subject?
3. Decisiveness: Did it avoid redundant steps and make clear, timely judgments?

TRAJECTORY DATA:
{trajectory_json}

Provide your evaluation in the following JSON format:
{{
    "reasoning_score": <float strictly between 0.01 and 0.99>,
    "strategy_score": <float strictly between 0.01 and 0.99>,
    "decisiveness_score": <float strictly between 0.01 and 0.99>,
    "overall_qualitative_score": <float strictly between 0.01 and 0.99>,
    "criticism": "<brief explanation of failures>",
    "commendation": "<brief explanation of strengths>"
}}
"""

def main():
    parser = argparse.ArgumentParser(description="SentinelOps LLM-as-a-Judge")
    parser.add_argument("--state-file", help="Path to episode state JSON")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)

    # Load trajectory
    try:
        if args.state_file:
            with open(args.state_file, "r") as f:
                state_data = json.load(f)
        else:
            # Try to read from local server if no file provided
            import requests
            resp = requests.get("http://localhost:7860/state")
            state_data = resp.json().get("episode", {})
    except Exception as e:
        print(f"Error loading state: {e}")
        sys.exit(1)

    # Extract history for the judge
    history = state_data.get("action_history", [])
    if not history:
        print("No action history found in state.")
        return

    # Clean history for prompt
    cleaned_history = []
    for h in history:
        cleaned_history.append({
            "step": h.get("step"),
            "action": h.get("action_type"),
            "payload": h.get("payload"),
            "reasoning": h.get("reasoning", "N/A")
        })

    client = OpenAI(base_url=API_BASE_URL, api_key=os.environ.get("API_KEY", HF_TOKEN))

    print(f"--- Sentinel-Judge Evaluation ---")
    print(f"Model: {MODEL_NAME}")
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional security operations judge."},
                {"role": "user", "content": JUDGE_PROMPT.format(trajectory_json=json.dumps(cleaned_history, indent=2))}
            ],
            temperature=0.1
        )
        
        response_text = completion.choices[0].message.content
        # Extract JSON
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            for k in ["reasoning_score", "strategy_score", "decisiveness_score", "overall_qualitative_score"]:
                if k in result:
                    result[k] = round(max(0.01, min(0.99, float(result[k]))), 4)
            print(json.dumps(result, indent=2))
        else:
            print("Judge provided non-JSON feedback:")
            print(response_text)
            
    except Exception as e:
        print(f"Judging failed: {e}")

if __name__ == "__main__":
    main()
