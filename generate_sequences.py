import json
import base64
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TASKS_DIR = Path("tasks")
SEQUENCES_DIR = Path("assets/sequences")
FRAMES_DIR = Path("assets/frames")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_sequences")

def create_frame_image(task_id: str, frame_data: dict, out_path: Path):
    """Generate a 640x480 frame with overlay information based on the JSON description."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Base background color: slightly darker if anomaly is present to simulate event change?
    # Or just load the static frame and draw on top of it.
    camera_id = frame_data["camera_id"]
    anomaly = frame_data.get("anomaly_present", False)
    
    # Try to load the base static frame
    base_filename = "cam01_normal.png"
    if "cam-01" in camera_id and anomaly: base_filename = "cam01_anomaly.png"
    elif "cam-02" in camera_id: base_filename = "cam02_suspicious.png"
    elif "cam-03" in camera_id and not anomaly: base_filename = "cam03_corridor.png"
    elif "cam-03" in camera_id and anomaly: base_filename = "cam03_intruder.png"
    elif "cam-04" in camera_id: base_filename = "cam04_entrance.png"
    
    try:
        base_img_path = FRAMES_DIR / base_filename
        if base_img_path.exists():
            img = Image.open(base_img_path).convert("RGB")
        else:
            img = Image.new("RGB", (640, 480), color=(30, 30, 40))
    except Exception:
        img = Image.new("RGB", (640, 480), color=(30, 30, 40))
        
    # Resize to standard
    img = img.resize((640, 480))
    draw = ImageDraw.Draw(img)
    
    # Try to load a simple font, fallback to default
    try:
        font_large = ImageFont.truetype("Arial.ttf", 24)
        font_small = ImageFont.truetype("Arial.ttf", 16)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
    # Draw top bar (Camera ID and Timestamp)
    draw.rectangle([(0, 0), (640, 40)], fill=(0, 0, 0, 200))
    draw.text((10, 10), f"CAMERA: {camera_id.upper()}", fill=(255, 255, 255), font=font_large)
    timestamp = frame_data.get("timestamp", "00:00:00")
    draw.text((400, 10), f"TIME: {timestamp}", fill=(255, 255, 255), font=font_large)
    
    # Draw Context description at the bottom
    desc = frame_data.get("description", "")
    import textwrap
    wrapped_desc = textwrap.fill(desc, width=70)
    bg_height = 480 - (20 * len(wrapped_desc.split('\n')) + 20)
    draw.rectangle([(0, bg_height), (640, 480)], fill=(0, 0, 0, 180))
    draw.text((10, bg_height + 10), wrapped_desc, fill=(200, 255, 200), font=font_small)
    
    # Draw alert indicator if anomaly
    if anomaly:
        region = frame_data.get("anomaly_region", "center")
        draw.text((10, 50), f"[!] ANOMALY DETECTED IN REGION: {region.upper()}", fill=(255, 50, 50), font=font_large)
        
    img.save(out_path, format="PNG")
    
def generate_all():
    tasks = list(TASKS_DIR.rglob("*.json"))
    logger.info("Found %d tasks. Generating frames...", len(tasks))
    count = 0
    for task_path in tasks:
        with open(task_path) as fp:
            data = json.load(fp)
            
        task_id = data.get("task_id")
        frames = data.get("frames", [])
        for frame in frames:
            frame_id = frame.get("frame_id")
            if not frame_id: continue
            out_file = SEQUENCES_DIR / task_id / f"{frame_id}.png"
            create_frame_image(task_id, frame, out_file)
            count += 1
            
    logger.info("Successfully generated %d sequences files.", count)

if __name__ == "__main__":
    generate_all()
