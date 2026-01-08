import os
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil

BASE_DIR = Path(__file__).resolve().parent
MATERIALS_DIR = BASE_DIR / "materials"
VIDEO_NAME = "backflip.mp4" 
VIDEO_PATH = MATERIALS_DIR / VIDEO_NAME
PROMPT_TEXT = "person" 

SESSION_DIR = BASE_DIR / f"proc_img_out"
FRAMES_DIR = SESSION_DIR / "frames"
OUTPUT_FOLDER = SESSION_DIR / "transparent_pngs"
FINAL_VIDEO = BASE_DIR / f"{Path(VIDEO_NAME).stem}_isolated.mov"

def setup_directories():
    # if SESSION_DIR.exists():
    #     print(f"Limpando sessão anterior em {SESSION_DIR}...")
    #     shutil.rmtree(SESSION_DIR)
    
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def extract_frames():
    print(f"Extraindo frames de {VIDEO_NAME}...")
    cmd = f"ffmpeg -i {VIDEO_PATH} -vf scale=1280:-1 -q:v 2 -start_number 0 {FRAMES_DIR}/%05d.jpg -y"
    subprocess.run(cmd, shell=True, check=True)

def process_video_frames():
    print("Inicializando SAM 3 Image")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    frame_files = sorted(list(FRAMES_DIR.glob("*.jpg")))
    print(f"Processando {len(frame_files)} frames...")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, frame_path in enumerate(tqdm(frame_files)):
            try:
                image = Image.open(frame_path).convert("RGB")
                img_np = np.array(image)

                inference_state = processor.set_image(image)
                output = processor.set_text_prompt(state=inference_state, prompt=PROMPT_TEXT)

                masks = output["masks"]
                if masks is None or masks.shape[0] == 0:
                    continue

                mask = masks[0].cpu().numpy().squeeze()

                alpha = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                alpha[mask > 0] = 255

                rgba_img = np.dstack((img_np, alpha))
                out_filename = OUTPUT_FOLDER / f"frame_{idx:05d}.png"
                Image.fromarray(rgba_img, "RGBA").save(out_filename)

                del inference_state, output, masks, mask, alpha, rgba_img

            except Exception as e:
                print(f"Erro no frame {idx}: {e}")
                continue

PNG_FOLDER = BASE_DIR / f"proc_img_out" / "transparent_pngs"
TEMP_SEQ = BASE_DIR / "temp_final_alpha"
OUTPUT_VIDEO = BASE_DIR / "extracted_subject.mov"

def assemble_video():

    if TEMP_SEQ.exists(): shutil.rmtree(TEMP_SEQ)
    TEMP_SEQ.mkdir(parents=True)

    png_files = sorted(list(PNG_FOLDER.glob("frame_*.png")))
    if not png_files:
        print(f"ERRO: Nenhum PNG encontrado em {PNG_FOLDER}")
        return

    with Image.open(png_files[0]) as img:
        print(f"-> Verificando {png_files[0].name} | Modo: {img.mode}")
        if img.mode != 'RGBA':
            print("ERRO: Os PNGs não têm Alpha. O problema está no script de extração.")
            return

    print("-> Criando sequência de frames limpa...")
    for i, file_path in enumerate(png_files):
        shutil.copy(file_path, TEMP_SEQ / f"frame_{i:05d}.png")

    v_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=rgba"
    
    cmd = [
        "ffmpeg", "-framerate", "30",
        "-i", str(TEMP_SEQ / "frame_%05d.png"),
        "-vf", v_filter,
        "-c:v", "png",
        str(OUTPUT_VIDEO), "-y"
    ]

    print("-> Renderizando vídeo com codec PNG...")
    try:
        subprocess.run(cmd, check=True)
        print(f"SUCESSO! Vídeo gerado em: {OUTPUT_VIDEO}")
    except subprocess.CalledProcessError as e:
        print(f"ERRO NO FFMPEG: {e}")
    finally:
        if TEMP_SEQ.exists(): shutil.rmtree(TEMP_SEQ)

INPUT_MOV = BASE_DIR / "extracted_subject.mov"
OUTPUT_GIF = BASE_DIR / "iphone_gif_transparent.gif"

def convert_to_final_gif():
    if not INPUT_MOV.exists():
        return

    v_filter = (
        "fps=15,scale=480:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=reserve_transparent=1[p];"
        "[s1][p]paletteuse=alpha_threshold=128"
    )

    cmd = [
        "ffmpeg", "-i", str(INPUT_MOV),
        "-vf", v_filter,
        "-loop", "0",
        "-final_delay", "0",
        str(OUTPUT_GIF), "-y"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"GIF gerado: {OUTPUT_GIF}")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    setup_directories()
    extract_frames()
    process_video_frames()
    assemble_video()
    convert_to_final_gif()
    