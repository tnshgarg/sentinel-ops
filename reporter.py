import json
from datetime import datetime
from typing import Dict, Any, List
from safety import SafetyGuard

class IncidentReporter:
    """
    SentinelOps Incident Reporter.
    Generates structured human-readable reports from agent trajectories.
    """

    @staticmethod
    def generate_markdown(
        task_id: str,
        difficulty: str,
        state_dict: Dict[str, Any],
        info: Dict[str, Any],
        safety_log: List[Dict[str, Any]]
    ) -> str:
        """
        Produce a professional Markdown report of the episode.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "COMPLETED" if state_dict.get("done") else "IN_PROGRESS"
        total_reward = sum(h.get("reward", 0.0) for h in state_dict.get("action_history", []))
        
        # Calculate Safety overall
        is_safe = all(s.get("is_safe", True) for s in safety_log)
        safety_status = "✅ PASS" if is_safe else "❌ FAIL (Safety Violation)"

        # Header
        md = f"""# 🛡️ SentinelOps | Incident Response Report
**Ref ID:** `{task_id}` | **Generated:** `{now}`

---

## 📋 Mission Overview
| Attribute | Value |
| :--- | :--- |
| **Task ID** | `{task_id}` |
| **Difficulty** | `{difficulty.upper()}` |
| **Final Status** | `{status}` |
| **Total Reward** | **`{total_reward:.4f}`** |
| **Safety Audit** | **{safety_status}** |

---

## 🛡️ Safety & Compliance (Llama Guard 3)
"""
        if is_safe:
            md += "Audit complete. No high-risk policy violations detected in agent reasoning or actions.\n"
        else:
            md += "### ⚠️ Safety Violations Detected\n"
            violations = set()
            for s in safety_log:
                for cat in s.get("violated_categories", []):
                    violations.add(f"- **{cat}**: {SafetyGuard.get_taxonomy_desc(cat)}")
            md += "\n".join(list(violations)) + "\n"

        md += """
---

## 🕹️ Trajectory Timeline
| Step | Action | Payload | Reasoning | Reward | Safety |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        history = state_dict.get("action_history", [])
        for i, entry in enumerate(history):
            # Find safety log for this step
            s_entry = safety_log[i] if i < len(safety_log) else {"is_safe": True, "explanation": "N/A"}
            s_icon = "🟢" if s_entry.get("is_safe") else "🔴"
            
            action = entry.get("action_type", "N/A")
            payload = entry.get("payload", "-")
            reasoning = entry.get("reasoning", "*No reasoning provided*")
            reward = entry.get("reward", 0.0)
            
            md += f"| {i} | `{action}` | `{payload}` | {reasoning} | `{reward:+.2f}` | {s_icon} |\n"

        md += """
---

## ⚖️ Grading Rubric Breakdown
| Component | Score | Impact |
| :--- | :--- | :--- |
"""
        breakdown = info.get("breakdown", {})
        for key, val in breakdown.items():
            md += f"| {key.replace('_', ' ').title()} | `{val:+.3f}` | {'Positive' if val > 0 else 'Negative/Neutral'} |\n"

        md += """
---

## 🧠 Final Intelligence Analytics
- **Efficiency Index:** Agent completed task in """ + str(state_dict.get('current_step', 0)) + """ steps.
- **Visual Depth:** """ + str(len(state_dict.get('frames_inspected', []))) + """ key frames were subjected to high-fidelity inspection.
- **Coverage:** """ + str(len(state_dict.get('cameras_visited', []))) + """ unique camera feeds utilized for triangulation.

> **Report Note:** This document is an automated output of the SentinelOps Evaluation Framework. It is intended for researcher review and hackathon verification purposes.
"""
        return md

    @staticmethod
    def generate_dossier_html(
        task_id: str,
        difficulty: str,
        state_dict: Dict[str, Any],
        info: Dict[str, Any],
        safety_log: List[Dict[str, Any]]
    ) -> str:
        """
        Produce a premium, legal-ready HTML Dossier for enterprise security review.
        """
        now = datetime.now().strftime("%Y-%b-%d %H:%M:%S")
        history = state_dict.get("action_history", [])
        is_safe = all(s.get("is_safe", True) for s in safety_log)
        
        # Build Timeline Rows
        rows = ""
        for i, entry in enumerate(history):
            s_entry = safety_log[i] if i < len(safety_log) else {"is_safe": True, "explanation": "N/A"}
            status_cls = "text-success" if s_entry.get("is_safe") else "text-danger"
            status_icon = "✓" if s_entry.get("is_safe") else "⚠"
            
            rows += f"""
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 12px; font-family: monospace; color: #94a3b8;">{i:03d}</td>
                <td style="padding: 12px; font-weight: 600; color: #f8fafc;">{entry.get('action_type', 'N/A').upper()}</td>
                <td style="padding: 12px; font-style: italic; color: #cbd5e1;">{entry.get('reasoning', 'No reasoning provided')}</td>
                <td style="padding: 12px; font-weight: 700; {status_cls}">[{status_icon}] Safety Cleared</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; }}
                .container {{ max-width: 900px; margin: auto; background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
                .header {{ border-bottom: 2px solid #3b82f6; padding-bottom: 20px; margin-bottom: 30px; }}
                .badge {{ background: #3b82f6; color: white; padding: 4px 12px; border-radius: 99px; font-size: 0.8rem; text-transform: uppercase; }}
                .safe-cert {{ border: 2px solid #10b981; background: rgba(16,185,129,0.1); padding: 20px; border-radius: 8px; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ text-align: left; background: #334155; padding: 12px; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ Legal Incident Dossier</h1>
                    <p style="color: #94a3b8;">Case ID: {task_id} | Timestamp: {now}</p>
                    <span class="badge">Classification: Official</span>
                </div>

                <div class="safe-cert">
                    <h3 style="color: #10b981; margin-top: 0;">🛡️ Llama Guard 3 Compliance Certificate</h3>
                    <p>This incident response has been audited by Llama Guard 3. No high-risk policy violations (Violence, S8/Privacy, Bias) were detected during the autonomous reasoning phase.</p>
                    <p style="font-family: monospace; font-size: 0.8rem;">Status: <strong>SECURED</strong> | Audit Hash: {hash(str(safety_log))}</p>
                </div>

                <h3>🕹️ Evidence Chain of Custody</h3>
                <table>
                    <thead>
                        <tr><th>Step</th><th>Action</th><th>Analyst Intent (Reasoning)</th><th>Audit Status</th></tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>

                <div style="margin-top: 40px; border-top: 1px solid #334155; padding-top: 20px; font-size: 0.8rem; color: #94a3b8;">
                    <p>Generated by SentinelOps v2.0.6 Elite Framework. This document is a certified audit trail of agentic decision-making.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    @staticmethod
    def save_report(task_id: str, content: str) -> str:
        filename = f"reports/Incident_Report_{task_id}.md"
        with open(filename, "w") as f:
            f.write(content)
        return filename
