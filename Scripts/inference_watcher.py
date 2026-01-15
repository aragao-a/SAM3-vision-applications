import torch
import numpy as np
import cv2
import yaml
import subprocess
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

os.environ["QT_QPA_PLATFORM"] = "offscreen"

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
VIDEO_DIR = SCRIPT_DIR / "materials" / "1080p"
RUNS_DIR = SCRIPT_DIR / "runs"

PROMPT_CONFIG = {
    "car window": {"color": (0, 140, 255), "label": "Window"},
    "tires": {"color": (255, 0, 0), "label": "Tire"},
    "bumpers": {"color": (65, 169, 76), "label": "Bumper"},
}

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def setup_dirs():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = RUNS_DIR / f"batch_{run_id}"
    raw_dir = session_dir / "snapshots"
    out_dir = session_dir / "inference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, out_dir

def extract_frames(cfg, raw_dir):
    video_name = cfg['simulation']['video_name']
    interval = cfg['simulation']['frame_snap_interval']
    res = cfg['simulation']['output_resolution']
    video_path = VIDEO_DIR / video_name
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video nao encontrado em {video_path}")

    filter_str = f"select='not(mod(n,{interval}))',scale={res[0]}:{res[1]}"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", filter_str, "-fps_mode", "vfr", "-q:v", "2",
        str(raw_dir / "frame_%05d.jpg"), "-y"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def apply_overlay(frame, mask, color):
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

def run_process():
    config = load_config()
    raw_dir, out_dir = setup_dirs()
    
    extract_frames(config, raw_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    frame_files = sorted(list(raw_dir.glob("*.jpg")))

    with torch.autocast(device, dtype=torch.bfloat16):
        for frame_path in tqdm(frame_files, desc="Processando"):
            frame_cv = cv2.imread(str(frame_path))
            img_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            state = processor.set_image(pil_img)

            for prompt, p_cfg in PROMPT_CONFIG.items():
                out = processor.set_text_prompt(state=state, prompt=prompt)
                
                if out["masks"] is not None:
                    for i, mask_tensor in enumerate(out["masks"]):
                        if out["scores"][i] < config['model_params']['confidence_threshold']:
                            continue
                        
                        mask = mask_tensor.cpu().numpy().squeeze()
                        box = out["boxes"][i].cpu().numpy().astype(int)
                        frame_cv = apply_overlay(frame_cv, mask, p_cfg["color"])
                        cv2.putText(frame_cv, p_cfg["label"], (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_cfg["color"], 2)

            cv2.imwrite(str(out_dir / f"processed_{frame_path.name}"), frame_cv)

if __name__ == "__main__":
    run_process()