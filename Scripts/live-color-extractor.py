import torch
import numpy as np
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil
from datetime import datetime
from img_to_vid import  concatenate_frames_to_video

BASE_DIR = Path(__file__).resolve().parent
VIDEO_NAME = "GT7-Easy1-30.mp4"
VIDEO_PATH = BASE_DIR / "materials" / "1080p" / VIDEO_NAME

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_BASENAME = Path(VIDEO_NAME).stem
RUN_ID = f"{VIDEO_BASENAME}_{RUN_TIMESTAMP}"

SESSION_DIR = BASE_DIR / "proc_img_out" / RUN_ID
RAW_FRAMES_DIR = SESSION_DIR / "raw_frames"
SEGMENTED_DIR = SESSION_DIR / "segmented_frames"

OPACITY = 0.4 
PROMPT_CONFIG = {
    "car window": {"color": (0, 140, 255), "label": "Window"},
    "tires": {"color": (255, 0, 0), "label": "Tire"},
    "bumpers": {"color": (65, 169, 76), "label": "Bumper"},
}

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_directories():
    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

def extract_frames():
    cmd = f"ffmpeg -i {VIDEO_PATH} -vf scale=1920:1080 -q:v 2 -start_number 0 {RAW_FRAMES_DIR}/frame_%05d.jpg -y"
    subprocess.run(cmd, shell=True, check=True)

def apply_overlay(frame, mask, color):
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 1.0 - OPACITY, overlay, OPACITY, 0)

def process_and_colorize():
    device = select_device()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    frame_files = sorted(list(RAW_FRAMES_DIR.glob("*.jpg")))
    with torch.autocast(device, dtype=torch.bfloat16):
        for idx, frame_path in enumerate(tqdm(frame_files)):
            try:
                frame_cv = cv2.imread(str(frame_path))
                img_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                inference_state = processor.set_image(pil_img)

                for prompt_text, config in PROMPT_CONFIG.items():
                    output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
                    
                    masks = output["masks"]
                    scores = output["scores"]
                    boxes = output["boxes"]

                    if masks is not None:
                        for i, mask_tensor in enumerate(masks):
                            if scores[i] < 0.5: continue

                            mask = mask_tensor.cpu().numpy().squeeze()
                            box = boxes[i].cpu().numpy().astype(int)

                            frame_cv = apply_overlay(frame_cv, mask, config["color"])
                            
                            cv2.putText(frame_cv, config["label"], (box[0], box[1] - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, config["color"], 2)

                out_filename = SEGMENTED_DIR / f"segmented_{idx:05d}.jpg"
                cv2.imwrite(str(out_filename), frame_cv)

            except Exception as e:
                continue

if __name__ == "__main__":
    setup_directories()
    extract_frames()
    process_and_colorize()
    concatenate_frames_to_video(RUN_ID)
    print(f"Feito: {RUN_ID}")