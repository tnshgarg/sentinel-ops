"""
SentinelOps — Sequence Frame Generator.

Generates per-task frame sequences from static base camera images.
Applies realistic CCTV HUD overlays (camera ID bar, timestamp) WITHOUT
exposing anomaly information in the visual — anomaly labels must only
exist in the JSON metadata and text context, never burned into the image.

This ensures a Vision-Language Model agent must actually *reason* about
the scene rather than just reading text from the frame.
"""

import json
import logging
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

TASKS_DIR = Path("tasks")
SEQUENCES_DIR = Path("assets/sequences")
FRAMES_DIR = Path("assets/frames")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_sequences")


def _get_camera_color(camera_id: str) -> tuple[int, int, int]:
    """Deterministically generate a subtle HUD bar color based on the camera string."""
    import hashlib
    val = int(hashlib.md5(camera_id.encode()).hexdigest(), 16)
    return (val % 80 + 20, (val // 80) % 80 + 20, (val // 6400) % 80 + 20)


def _get_base_image(camera_id: str, anomaly_present: bool) -> Image.Image:
    """
    Dynamically resolve the best base frame from assets/frames without hardcoding.
    Searches for files matching the camera ID prefix.
    """
    cam_prefix = camera_id.replace("-", "").lower()
    
    matches = list(FRAMES_DIR.glob(f"{cam_prefix}*.png"))
    
    best_match = None
    if matches:
        if anomaly_present:
            # Prefer images explicitly named anomaly/suspicious if present
            anomaly_imgs = [m for m in matches if any(w in m.name.lower() for w in ["anomaly", "suspicious", "intruder"])]
            best_match = anomaly_imgs[0] if anomaly_imgs else matches[0]
        else:
            # Prefer normal images
            normal_imgs = [m for m in matches if any(w in m.name.lower() for w in ["normal", "corridor", "entrance"])]
            best_match = normal_imgs[0] if normal_imgs else matches[0]

    if best_match and best_match.exists():
        img = Image.open(best_match).convert("RGB")
    else:
        # Procedural fallback frame if no asset matches the camera
        img = Image.new("RGB", (640, 480), color=(15, 15, 20))

    return img.resize((640, 480))


def _apply_cctv_hud(img: Image.Image, camera_id: str, timestamp: str,
                     frame_index: int, total_frames: int) -> Image.Image:
    """
    Draw a clean CCTV-style HUD overlay.

    Includes ONLY:
    - Camera ID label (top-left)
    - Timestamp (top-right)
    - Recording indicator dot
    - Frame counter
    - Subtle scanline effect

    Does NOT include any anomaly labels, risk text, or detection markers.
    """
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font_hud = ImageFont.truetype("Arial.ttf", 18)
        font_small = ImageFont.truetype("Arial.ttf", 13)
    except IOError:
        font_hud = ImageFont.load_default()
        font_small = ImageFont.load_default()

    bar_color = _get_camera_color(camera_id)

    # Top bar background
    draw.rectangle([(0, 0), (640, 32)], fill=(0, 0, 0))

    # Camera ID — left side
    draw.text((8, 6), f"● {camera_id.upper()}", fill=(200, 200, 200), font=font_hud)

    # Recording indicator (red dot)
    draw.ellipse([(610, 8), (630, 28)], fill=(220, 30, 30))

    # Timestamp — right side
    ts_display = timestamp.split("T")[1][:8] if "T" in timestamp else timestamp
    draw.text((400, 6), ts_display, fill=(180, 180, 180), font=font_hud)

    # Bottom bar — frame counter only
    draw.rectangle([(0, 460), (640, 480)], fill=(0, 0, 0))
    draw.text((8, 462), f"FRAME {frame_index + 1}/{total_frames}", fill=(140, 140, 140), font=font_small)

    # Subtle border tint matching camera theme
    for i in range(2):
        draw.rectangle([(i, i), (639 - i, 479 - i)], outline=(*bar_color, 80))

    return img

def _apply_visual_mode(img: Image.Image, mode: str) -> Image.Image:
    """
    Apply high-fidelity visual transformations to test VLM robustness.
    Modes: night_vision, thermal, fog, normal.
    """
    if mode == "night_vision":
        # 1. Darken and Green-shift
        img = img.convert("L").convert("RGB")  # Grayscale first
        overlay = Image.new("RGB", img.size, (0, 255, 60))
        img = Image.blend(img, overlay, 0.2)
        
        # 2. Add 'Film Grain' / Sensor Noise
        import numpy as np
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
        # 3. Brightness/Contrast adjust
        img = ImageEnhance.Brightness(img).enhance(0.8)
        img = ImageEnhance.Contrast(img).enhance(1.4)
        
    elif mode == "thermal":
        # 1. Luminance map to pseudo-color
        img = img.convert("L")
        # Custom Thermal Palette mapping (Simplified: Blue -> Red -> Yellow)
        import numpy as np
        arr = np.array(img)
        res = np.zeros((*arr.shape, 3), dtype=np.uint8)
        res[:, :, 0] = arr  # R
        res[:, :, 1] = (arr // 2)  # G
        res[:, :, 2] = (255 - arr) // 2  # B
        img = Image.fromarray(res)
        img = ImageEnhance.Contrast(img).enhance(1.8)
        
    elif mode == "fog":
        # Atmospheric haze overlay
        overlay = Image.new("RGB", img.size, (200, 205, 210))
        img = Image.blend(img, overlay, 0.4)
        img = img.filter(ImageFilter.GaussianBlur(1))
        
    return img

def _apply_temporal_variation(img: Image.Image, frame_index: int, anomaly_present: bool) -> Image.Image:
    """
    Apply subtle visual variation between frames to simulate real temporal progression.
    Different frames get slightly different brightness/contrast — NOT anomaly labels.
    """
    # Slight brightness shift per frame to simulate lighting changes
    enhancer = ImageEnhance.Brightness(img)
    brightness_factor = 1.0 + (frame_index % 3 - 1) * 0.04  # ±4% shift
    img = enhancer.enhance(brightness_factor)

    # If anomaly is present, slightly increase contrast (subtle visual cue, not text)
    if anomaly_present:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.06)

    return img


def create_frame_image(task_id: str, frame_data: dict, frame_index: int,
                       total_frames: int, out_path: Path):
    """Generate a clean CCTV frame for a task sequence."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    camera_id = frame_data["camera_id"]
    anomaly = frame_data.get("anomaly_present", False)
    timestamp = frame_data.get("timestamp", "00:00:00")

    # Load base image
    img = _get_base_image(camera_id, anomaly)
    
    # 1. Apply High-Fidelity Visual Mode (Night/Thermal/Fog)
    v_mode = frame_data.get("visual_mode", "normal")
    img = _apply_visual_mode(img, v_mode)

    # 2. Apply temporal variation (subtle brightness/contrast shifts)
    img = _apply_temporal_variation(img, frame_index, anomaly)

    # 3. Apply clean CCTV HUD (NO anomaly text)
    img = _apply_cctv_hud(img, camera_id, timestamp, frame_index, total_frames)

    img.save(out_path, format="PNG")


def generate_all():
    """Regenerate all task sequence frames."""
    tasks = list(TASKS_DIR.rglob("*.json"))
    logger.info("Found %d tasks. Generating clean CCTV frames...", len(tasks))
    count = 0

    for task_path in tasks:
        with open(task_path) as fp:
            data = json.load(fp)

        task_id = data.get("task_id")
        frames = data.get("frames", [])
        total_frames = len(frames)

        for idx, frame in enumerate(frames):
            frame_id = frame.get("frame_id")
            if not frame_id:
                continue
            out_file = SEQUENCES_DIR / task_id / f"{frame_id}.png"
            create_frame_image(task_id, frame, idx, total_frames, out_file)
            count += 1

    logger.info("Successfully generated %d clean CCTV frames.", count)


if __name__ == "__main__":
    generate_all()
