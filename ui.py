import streamlit as st
import base64
import requests
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="SentinelOps Agent Visualizer", layout="wide")

st.title("🛡️ SentinelOps - Human/Agent Visualizer")
st.markdown("Use this interface to manually test the SentinelOps environment. It mirrors exactly what the AI sees and verifies the backend grading logic is flawless.")

# The EnvClient class inline for portability
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id=None):
        payload = {"task_id": task_id} if task_id else {}
        resp = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        return resp.json()

    def step(self, action_type, payload=None):
        body = {"action_type": action_type, "payload": payload, "confidence": 1.0}
        resp = self.session.post(f"{self.base_url}/step", json=body, timeout=30)
        return resp.json()

    def state(self):
        resp = self.session.get(f"{self.base_url}/state", timeout=30)
        return resp.json()

    def tasks(self):
        resp = self.session.get(f"{self.base_url}/tasks", timeout=30)
        return resp.json()

client = EnvClient("http://localhost:7860")

# Initialize session state
if "obs" not in st.session_state:
    st.session_state.obs = None
if "info" not in st.session_state:
    st.session_state.info = None
if "terminated" not in st.session_state:
    st.session_state.terminated = False

# Sidebar config
st.sidebar.header("Control Panel")

try:
    available_tasks = client.tasks()
    task_options = {t["title"]: t["task_id"] for t in available_tasks}
    selected_task_title = st.sidebar.selectbox("Select Task", list(task_options.keys()))
    selected_task_id = task_options[selected_task_title]
except Exception as e:
    st.sidebar.error("Cannot connect to SentinelOps Server. Is it running on port 7860?")
    st.stop()

if st.sidebar.button("🚨 Reset Environment"):
    try:
        data = client.reset(selected_task_id)
        st.session_state.obs = data["observation"]
        st.session_state.info = data.get("info", {})
        st.session_state.terminated = False
        st.sidebar.success("Environment Reset!")
        st.rerun()
    except Exception as e:
        st.sidebar.error("Server Reset Failed.")

# Main Display
obs = st.session_state.obs
if obs is None:
    st.info("👈 Click **Reset Environment** in the sidebar to start.")
else:
    # 2-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Camera Feed")
        
        # Decode base64 image
        b64_img = obs.get("frame_b64", "")
        if b64_img:
            image_data = base64.b64decode(b64_img)
            image = Image.open(BytesIO(image_data))
            st.image(image, use_container_width=True, caption=f"Task: {obs.get('task_id')} | Camera: {obs.get('camera_id')}")
        else:
            st.warning("No image data received.")
            
        st.markdown("### 📝 Scene Context")
        st.info(obs.get("context", "No context provided."))

    with col2:
        st.subheader("State Metadata")
        st.json({
            "Step": obs.get("step"),
            "Alert Level": obs.get("alert_level"),
            "Cameras Visited": obs.get("metadata", {}).get("cameras_visited", []),
            "Current Reward": st.session_state.info.get("cumulative_reward", 0.0)
        })
        
        if st.session_state.terminated:
            st.success("✅ **Episode Completed**")
            st.write(f"**Final Score:** {st.session_state.info.get('cumulative_reward', 0.0)}")
            st.json(st.session_state.info)
        else:
            st.subheader("Take Action")
            actions = obs.get("available_actions", [])
            action_choice = st.selectbox("Action Type", actions)
            
            payload = ""
            if action_choice == "switch_camera":
                valid_cams = obs.get("metadata", {}).get("camera_ids", ["cam-01"])
                payload = st.selectbox("Payload (Camera ID)", valid_cams)
            elif action_choice == "zoom_region":
                payload = st.selectbox("Payload (Region)", ["top-left", "top-right", "bottom-left", "bottom-right", "center", "left", "right"])
            elif action_choice == "classify_risk":
                payload = st.selectbox("Payload (Risk)", ["safe", "suspicious", "dangerous", "critical"])
                
            if st.button("▶ Step"):
                try:
                    res = client.step(action_choice, payload if payload else None)
                    st.session_state.obs = res["observation"]
                    st.session_state.info = res["info"]
                    st.session_state.terminated = res["terminated"] or res["truncated"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Action failed: {e}")
