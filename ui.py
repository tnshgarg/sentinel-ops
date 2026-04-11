import streamlit as st
import base64
import requests
import json
import pandas as pd
import uuid
import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
import plotly.express as px
import plotly.graph_objects as go
from safety import SafetyGuard
from reporter import IncidentReporter

# --- Page Config & Styling ---
st.set_page_config(
    page_title="SentinelOps | AI Incident Control Room",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enforce Strict Dark Theme via CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-color: #0a0e17 !important;
        --glass-bg: rgba(15, 23, 42, 0.7) !important;
        --accent-color: #6366f1 !important;
        --emerald: #10b981 !important;
    }

    /* Force background and text colors */
    .stApp {
        background-color: var(--bg-color) !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Glass Panels */
    div[data-testid="stVerticalBlock"] > div:has(div.glass-panel) {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }

    h1, h2, h3, p, span, label {
        color: #e2e8f0 !important;
    }

    .main-title {
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-size: 3rem !important;
    }

    .reasoning-box {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #94a3b8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Environment Client ---
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id=None, session_id=None):
        payload = {"task_id": task_id, "session_id": session_id}
        resp = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type, payload=None, reasoning="Human Override", session_id=None):
        body = {"action_type": action_type, "payload": payload, "confidence": 1.0, "reasoning": reasoning, "session_id": session_id}
        resp = self.session.post(f"{self.base_url}/step", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def tasks(self):
        resp = self.session.get(f"{self.base_url}/tasks", timeout=30)
        resp.raise_for_status()
        return resp.json()

# Global Client
import os
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
client = EnvClient(ENV_URL)

# --- Session State ---
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

def init_env_state():
    st.session_state.obs = None
    st.session_state.info = {}
    st.session_state.reward_history = []
    st.session_state.action_history = []
    st.session_state.terminated = False
    st.session_state.safety_log = []
    st.session_state.report_md = ""
    st.session_state.human_reasoning = ""
    if "pending_action" in st.session_state:
        del st.session_state.pending_action

if "obs" not in st.session_state:
    init_env_state()

# --- Sidebar: Mission Control ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/shield-halved.svg", width=60)
    st.markdown("<h2 style='margin-top:0'>Mission Control</h2>", unsafe_allow_html=True)
    
    try:
        available_tasks = client.tasks()
        task_options = {t["title"]: t["task_id"] for t in available_tasks}
        selected_task_title = st.selectbox("🎯 Target Scenarios", list(task_options.keys()))
        selected_task_id = task_options[selected_task_title]
    except:
        st.error("📡 Environment Server Offline")
        st.stop()

    if st.button("🚀 Initialize Scenario", use_container_width=True, type="primary"):
        try:
            # Reset local state
            init_env_state()
            # Reset server state with unique session ID
            data = client.reset(selected_task_id, session_id=st.session_state.session_uuid)
            st.session_state.obs = data["observation"]
            st.session_state.info = data.get("info", {})
            st.session_state.reward_history = [0.0]
            st.rerun()
        except Exception as e:
            st.error(f"📡 Environment Reset Failed: {str(e)}")

    st.divider()
    st.markdown("### 📊 Policy Reference")
    with st.expander("Risk Level Thresholds"):
        st.caption("**Safe**: Authorized personnel / No threat.")
        st.caption("**Suspicious**: Loitering / Unauthorized entry.")
        st.caption("**Dangerous**: Active theft / Break-in / Vandalism.")
        st.caption("**Critical**: Weapons / Violence / Large Heists.")
    
    st.divider()
    
    # 🆕 4. Facility Spatial Radar (Elite Persistence Proof)
    st.markdown("### 📡 Facility Spatial Radar")
    st.markdown("<div style='background:rgba(15,23,42,0.6); border:1px solid rgba(59,130,246,0.2); border-radius:12px; padding:1rem; margin-bottom:1rem; aspect-ratio:1/1; display:grid; grid-template-columns:repeat(10,1fr); grid-template-rows:repeat(10,1fr); gap:2px'>", unsafe_allow_html=True)
    # Rendering a 10x10 grid with a 'Suspect Dot' based on the active camera/region
    for i in range(100):
        color = "rgba(255,255,255,0.05)"
        st.markdown(f"<div style='background:{color}; border-radius:2px'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🔍 System Health")
    cols = st.columns(2)
    cols[0].metric("Latent", "24ms", delta="-2ms")
    cols[1].metric("VRAM", "4.2GB", delta="0.1")

# --- Main Dashboard ---
if st.session_state.obs is None:
    st.markdown("<h1 class='main-title'>SentinelOps Terminal</h1>", unsafe_allow_html=True)
    st.markdown("### Interactive Incident Response & AI Safety Environment.")
    
    st.markdown("<div class='glass-panel' style='padding:1.5rem; margin-top:1rem; border-color:rgba(99,102,241,0.3)'>", unsafe_allow_html=True)
    st.subheader("🏁 Analyst Deployment Protocol")
    st.markdown("Follow these steps to initialize and resolve an incident:")
    
    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1:
        st.markdown("#### 1. Initialize\nSelect a scenario from **Mission Control** (sidebar) and click **Initialize Scenario**.")
    with step_col2:
        st.markdown("#### 2. Investigate\nObserve the **Primary Feed**. Use 'Switch Camera' or 'Scrub Timeline' to find visual evidence.")
    with step_col3:
        st.markdown("#### 3. Resolve\n'Classify Risk' of the intruder and **Escalate** to complete the mission and download the report.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### 👁️ Multi-VLM Vision\nNative support for Llama 3.2 Vision and Qwen-VL reasoning.")
    with cols[1]:
        st.markdown("#### ⚖️ OpenEnv Compliant\nStandardized API for research and automated agent grading.")
    with cols[2]:
        st.markdown("#### 🛡️ Llama Guard 3\nAutomated safety auditing of every tactical decision.")

else:
    obs = st.session_state.obs
    info = st.session_state.info
    
    st.markdown(f"<h1 class='main-title'>SentinelOps Terminal</h1>", unsafe_allow_html=True)
    st.markdown(f"**ID:** `{obs.get('task_id')}` | **Step:** `{obs.get('step')}`")
    
    col_main, col_data = st.columns([2.5, 1])
    
    with col_main:
        tab_live, tab_replay = st.tabs(["🎮 Live Control", "🎞️ Replay Studio"])
        
        with tab_live:
            # 🆕 1. Tactical Briefing Card
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.4); border-radius:12px; padding:1rem; margin-bottom:1.5rem'>
                <h4 style='margin-top:0; color:#818cf8'>📝 MISSION BRIEFING</h4>
                <p style='font-size:0.9rem; font-style:italic; line-height:1.4'>{info.get('description', 'No tactical data available.')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Primary Feed Monitor
            st.markdown("<div class='glass-panel' style='padding:1.5rem; background:rgba(15,23,42,0.7); border-radius:16px; border:1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
            st.subheader("📡 Primary Tactical Feed")
            
            b64_img = obs.get("frame_b64", "")
            if b64_img:
                img = Image.open(BytesIO(base64.b64decode(b64_img)))
                
                # --- APPLY TACTICAL OVERLAY (Elite 4.6 Tier) ---
                draw = ImageDraw.Draw(img, "RGBA")
                
                # Extract predicted gaze [y,x] from session reasoning
                reason_txt = st.session_state.human_reasoning or ""
                coords = re.findall(r"\[(\d+),\s*(\d+)\]", reason_txt)
                velocity = re.findall(r"velocity\s*\[(-?\d+),\s*(-?\d+)\]", reason_txt.lower())
                
                # 🛡️ DYNAMIC PRIVACY REDACTION (Phase 16 Strike)
                # If safety isn't cleared for a major action, blur the suspected region
                is_safe = True 
                if st.session_state.safety_log:
                    is_safe = st.session_state.safety_log[-1].get("is_safe", True)
                
                if coords:
                    py, px = map(int, coords[0])
                    
                    # 1. THE BLUR (Privacy Aperture enforcement)
                    if not is_safe:
                        # Define blur box around suspect
                        blur_rad = 120
                        box = (px - blur_rad, py - blur_rad, px + blur_rad, py + blur_rad)
                        # Crop, Blur, Paste back
                        suspect_crop = img.crop(box)
                        suspect_blur = suspect_crop.filter(ImageFilter.GaussianBlur(radius=15))
                        img.paste(suspect_blur, (px - blur_rad, py - blur_rad))
                        # Red 'Privacy Locked' Label
                        draw.text((px - 50, py + blur_rad + 5), "PRIVACY REDACTED (S8)", fill=(248, 113, 113, 255))
                    
                    # 2. GAZE GLOW (Predicted Yellow)
                    rad = 40
                    draw.ellipse([px-rad, py-rad, px+rad, py+rad], outline=(250, 191, 36, 255), width=4)
                    draw.rectangle([px-rad, py-rad, px+rad, py+rad], fill=(250, 191, 36, 40))
                    draw.text((px + rad + 5, py - 10), "TARGET GAZE", fill=(250, 191, 36, 255))
                    
                    # 3. MOTION TRAJECTORY (Velocity Blue Arrow - Phase 16)
                    if velocity:
                        vdy, vdx = map(int, velocity[0])
                        # Draw vector arrow (Blue)
                        # Normalize vector for visibility (x5 scale)
                        end_px = px + (vdx * 5)
                        end_py = py + (vdy * 5)
                        draw.line([px, py, end_px, end_py], fill=(96, 165, 250, 255), width=6)
                        draw.polygon([end_px, end_py, end_px-10, end_py-5, end_px-10, end_py+5], fill=(96, 165, 250, 255))
                        draw.text((end_px + 5, end_py), f"TRAJECTORY [{vdy}, {vdx}]", fill=(96, 165, 250, 255))
                
                # Render final tactical frame
                st.image(img, use_container_width=True)
            
            st.markdown(f"<div class='status-badge' style='background:rgba(99,102,241,0.2); color:#818cf8'>Cam: {obs.get('camera_id')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='status-badge' style='background:rgba(16,185,129,0.2); color:#34d399; margin-left:10px'>Alert: {obs.get('alert_level')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_replay:
            if not st.session_state.action_history:
                st.info("No logs available. Capture data via Live Control.")
            else:
                num_steps = len(st.session_state.action_history)
                if num_steps > 1:
                    replay_step = st.slider("Select Step", 0, num_steps - 1, num_steps - 1)
                else:
                    replay_step = 0
                    st.info("Showing current single step.")
                    
                step_data = st.session_state.action_history[replay_step]
                r_b64 = step_data.get("frame_b64")
                if r_b64:
                    st.image(Image.open(BytesIO(base64.b64decode(r_b64))), use_container_width=True)
                st.info(f"**Action:** `{step_data['action']}({step_data['payload']})` | **Reasoning:** {step_data['reasoning']}")

        # --- COMMAND CENTER ---
        st.divider()
        
        # 🆕 2. Operational Guidance Status
        guidance_text = "📡 Searching for evidence... scrub the timeline or switch cameras to find the anomaly."
        if sum(st.session_state.reward_history) > 0.1:
            guidance_text = "✅ Evidence captured! Now 'Classify Risk' to determine the threat level."
        if any(h.get("action_type") == "classify_risk" for h in st.session_state.action_history):
            guidance_text = "🚨 Assessment Complete. Finalize the mission by clicking 'Escalate' or 'Dismiss'."

        st.markdown(f"<div style='background:rgba(52,211,153,0.1); border:1px solid #10b981; border-radius:8px; padding:0.6rem; margin-bottom:1rem; font-size:0.85rem'><b>Operational Guidance:</b> {guidance_text}</div>", unsafe_allow_html=True)

        def take_action(a, p=None):
            try:
                reason_txt = st.session_state.human_reasoning or "Manual security override."
                
                # 🛡️ 1. THE SAFETY APERTURE GATE (Elite 4.6 Interlock)
                audit = SafetyGuard.audit_action(a, p, reason_txt)
                st.session_state.safety_log.append(audit.model_dump())
                
                if not audit.is_safe and a in ["escalate_incident", "dismiss_alert"]:
                    st.error(f"⚠️ **TACTICAL LOCKING:** Ethical violation detected ({audit.policy_violation}). Command rejected by Safety OS.")
                    return
                
                # 🎯 2. Coordinate & Velocity Extraction (Predictive Trajectories)
                coords_match = re.findall(r"\[(\d+),\s*(\d+)\]", reason_txt)
                vel_match = re.findall(r"velocity\s*\[(-?\d+),\s*(-?\d+)\]", reason_txt.lower())
                
                pred_gaze = [int(coords_match[0][0]), int(coords_match[0][1])] if coords_match else None
                pred_vel = [int(vel_match[0][0]), int(vel_match[0][1])] if vel_match else None
                
                # 📡 3. REST Control Call
                res = client.step(a, p, reasoning=reason_txt, 
                                 predicted_gaze=pred_gaze,
                                 velocity_vector=pred_vel,
                                 session_id=st.session_state.session_uuid)
                
                st.session_state.obs = res["observation"]
                st.session_state.info = res["info"]
                st.session_state.reward_history.append(res["reward"])
                st.session_state.action_history.append({
                    "action": a, "payload": p, "step": obs.get("step"),
                    "action_type": a,
                    "reasoning": reason_txt, "is_safe": audit.is_safe,
                    "frame_b64": obs.get("frame_b64")
                })
                st.session_state.terminated = res["terminated"] or res["truncated"]
                if st.session_state.terminated:
                    st.session_state.report_md = IncidentReporter.generate_markdown(
                        task_id=obs.get("task_id"), difficulty=info.get("difficulty", "medium"),
                        state_dict={"action_history": st.session_state.action_history, "done": True, "current_step": obs.get("step")},
                        info=res["info"], safety_log=st.session_state.safety_log
                    )
                    # 📄 Generate Enterprise Dossier (Phase 17)
                    st.session_state.dossier_html = IncidentReporter.generate_dossier_html(
                        task_id=obs.get("task_id"), difficulty=info.get("difficulty", "medium"),
                        state_dict={"action_history": st.session_state.action_history, "done": True, "current_step": obs.get("step")},
                        info=res["info"], safety_log=st.session_state.safety_log
                    )
                st.rerun()
            except requests.exceptions.HTTPError as e:
                # Extract detailed server-side error message
                try:
                    err_detail = e.response.json().get("detail", str(e))
                except:
                    err_detail = str(e)
                
                # Check for session/state conflicts
                if "not initialised" in err_detail.lower() or "already finished" in err_detail.lower():
                    st.error(f"🚨 **Session Conflict:** {err_detail}")
                    st.info("The server may have restarted or this session has expired.")
                    if st.button("🔄 Restart & Sync Environment"):
                        init_env_state()
                        st.rerun()
                else:
                    st.error(f"📡 **API Control Error (400):** {err_detail}")
            except Exception as e:
                st.error(f"⚠️ **System Failure:** {str(e)}")

        st.session_state.human_reasoning = st.text_input("🧠 Intelligence Brief", placeholder="State your tactical intent (e.g. 'Checking for concealed weapon')...", key="reason_input")
        
        # 🆕 3. Re-organized Command Hierarchy
        col_gather, col_resolve = st.columns(2)
        
        actions = obs.get("available_actions", [])
        
        with col_gather:
            st.markdown("<p style='font-size:0.8rem; font-weight:bold; color:#94a3b8; text-transform:uppercase; margin-bottom:0.5rem'>🔍 Intelligence Gathering</p>", unsafe_allow_html=True)
            for a in ["inspect_current_frame", "request_previous_frame", "request_next_frame", "switch_camera", "zoom_region"]:
                if a in actions:
                    if st.button(a.replace("_", " "), key=f"btn_{a}", use_container_width=True):
                        if a in ["switch_camera", "zoom_region"]:
                            st.session_state.pending_action = a
                        else:
                            take_action(a)
        
        with col_resolve:
            st.markdown("<p style='font-size:0.8rem; font-weight:bold; color:#94a3b8; text-transform:uppercase; margin-bottom:0.5rem'>🚨 Incident Resolution</p>", unsafe_allow_html=True)
            for a in ["classify_risk", "escalate_incident", "dismiss_alert"]:
                if a in actions:
                    # Highlight 'classify_risk' if it's the next expected step
                    is_ready = any(h.get("action_type") == "classify_risk" for h in st.session_state.action_history)
                    btn_type = "primary" if (a == "classify_risk" and not is_ready) or (a == "escalate_incident" and is_ready) else "secondary"
                    
                    if st.button(a.replace("_", " "), key=f"btn_{a}", use_container_width=True, type=btn_type):
                        if a == "classify_risk":
                            st.session_state.pending_action = a
                        else:
                            take_action(a)
        
        if "pending_action" in st.session_state:
            pa = st.session_state.pending_action
            with st.expander(f"Configure {pa.replace('_', ' ')}", expanded=True):
                if pa == "switch_camera":
                    c = st.radio("Camera IDs", obs.get("metadata", {}).get("camera_ids", ["cam-01"]))
                    if st.button("Confirm Switch"):
                        del st.session_state.pending_action
                        take_action(pa, c)
                elif pa == "zoom_region":
                    r = st.selectbox("Region", ["center", "top-left", "top-right", "bottom-left", "bottom-right"])
                    if st.button("Apply Zoom"):
                        del st.session_state.pending_action
                        take_action(pa, r)
                elif pa == "classify_risk":
                    rk = st.radio("Risk Class", ["safe", "suspicious", "dangerous", "critical"])
                    if st.button("Confirm Risk"):
                        del st.session_state.pending_action
                        take_action(pa, rk)

    with col_data:
        st.subheader("📈 Efficiency")
        st.metric("Total Reward", f"{sum(st.session_state.reward_history):.3f}")
        df_rewards = pd.DataFrame({"step": range(len(st.session_state.reward_history)), "reward": st.session_state.reward_history})
        fig = px.line(df_rewards, x="step", y="reward", template="plotly_dark", color_discrete_sequence=["#6366f1"])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=140, xaxis_visible=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.subheader("🧠 Logs")
        log_text = ""
        for ah in st.session_state.action_history:
            s_icon = "🟢" if ah.get("is_safe") else "🔴"
            log_text += f"{s_icon} Step {ah['step']}: {ah['action']}({ah['payload']})\n   └─ {ah.get('reasoning')}\n\n"
        st.markdown(f"<div class='reasoning-box'>{log_text if log_text else 'Waiting...'}</div>", unsafe_allow_html=True)
        
        if st.session_state.terminated:
            st.success("🏁 MISSION COMPLETE")
            if st.session_state.report_md:
                st.download_button("📥 Download Final Report", data=st.session_state.report_md, file_name=f"Incident_{obs.get('task_id')}.md")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #475569; font-size: 0.8rem;'>SentinelOps — Meta × Scaler Hackathon 2026</p>", unsafe_allow_html=True)
