import torch
import numpy as np
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from pathlib import Path
from tqdm import tqdm
import subprocess
from datetime import datetime

# Configurações de Caminho e Vídeo
BASE_DIR = Path(__file__).resolve().parent
VIDEO_NAME = "GT7-Easy1-30.mp4"
VIDEO_PATH = BASE_DIR / "materials" / "1080p" / VIDEO_NAME

# Setup de Saída
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = BASE_DIR / "proc_img_out" / f"task2_{RUN_TIMESTAMP}"
RAW_FRAMES_DIR = SESSION_DIR / "raw_frames"
SEGMENTED_DIR = SESSION_DIR / "segmented_frames"

# Configuração de Prompts e Cores (BGR para OpenCV)
OPACITY = 0.4
PROMPT_CONFIG = {
    "car window": {"color": (0, 140, 255), "label": "Window"},
    "tires": {"color": (255, 0, 0), "label": "Tire"},
    "bumpers": {"color": (65, 169, 76), "label": "Bumper"},
}

def setup_directories():
    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

def extract_frames_720p():
    """Usa ffmpeg para extrair frames já redimensionados para 720p"""
    print(f"-> Extraindo frames em 720p...")
    cmd = (f"ffmpeg -i {VIDEO_PATH} -vf scale=1280:720 -q:v 2 "
           f"-start_number 0 {RAW_FRAMES_DIR}/frame_%05d.jpg -y")
    subprocess.run(cmd, shell=True, check=True)

def apply_overlay(frame, mask, color):
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 1.0 - OPACITY, overlay, OPACITY, 0)

def run_inference_task2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Iniciando Inferência SAM3 em: {device.upper()}")
    
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    frame_files = sorted(list(RAW_FRAMES_DIR.glob("*.jpg")))
    
    # Autocast para bfloat16 (melhor performance na sua RTX Ada)
    with torch.autocast(device, dtype=torch.bfloat16):
        for idx, frame_path in enumerate(tqdm(frame_files, desc="Processando Frames")):
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
                        if scores[i] < 0.5: continue # Threshold de confiança

                        mask = mask_tensor.cpu().numpy().squeeze()
                        box = boxes[i].cpu().numpy().astype(int)

                        # Aplica a cor na máscara
                        frame_cv = apply_overlay(frame_cv, mask, config["color"])
                        
                        # Escreve a label (BUMPER ou TIRE)
                        cv2.putText(frame_cv, config["label"], (box[0], box[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config["color"], 2)

            # Salva o frame processado em 720p
            out_filename = SEGMENTED_DIR / f"out_{idx:05d}.jpg"
            cv2.imwrite(str(out_filename), frame_cv)

if __name__ == "__main__":
    setup_directories()
    extract_frames_720p()
    run_inference_task2()
    print(f"\n[OK] Processamento finalizado em {SESSION_DIR}")