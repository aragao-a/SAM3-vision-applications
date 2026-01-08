import subprocess
from pathlib import Path
import os
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "proc_img_out"
OUTPUT_FOLDER = BASE_DIR / "results"

RUN_FOLDER = "GT7-Easy3-30_20260108_114848"

INPUT_FOLDER = PROC_DIR / RUN_FOLDER / "segmented_frames"

FPS = 30
CRF = 23

def concatenate_frames_to_video():
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    print(f"-> Usando execução: {RUN_FOLDER}")

    if not INPUT_FOLDER.exists():
        print(f"Erro: Pasta de origem não encontrada: {INPUT_FOLDER}")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in INPUT_FOLDER.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not images:
        print(f"Erro: Nenhuma imagem encontrada em {INPUT_FOLDER}")
        return

    print(f"-> Localizados {len(images)} frames. Iniciando concatenação via FFmpeg...")

    OUTPUT_VIDEO = OUTPUT_FOLDER / f"{RUN_FOLDER}.mp4"
    
    cmd = [
        "ffmpeg",
        "-framerate", str(FPS),
        "-i", str(INPUT_FOLDER / "segmented_%05d.jpg"),
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(OUTPUT_VIDEO),
        "-y"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"-> ✅ SUCESSO! Vídeo gerado em: {OUTPUT_VIDEO}")
        print(f"-> Tamanho estimado: Médio (CRF {CRF})")
    except subprocess.CalledProcessError as e:
        print(f"ERRO NO FFMPEG: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    concatenate_frames_to_video()