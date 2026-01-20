import torch
import numpy as np
import cv2
import yaml
import ast
import time
import statistics
import csv
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from pathlib import Path
from tqdm import tqdm
import subprocess
from datetime import datetime
from typing import cast, Tuple, Dict, Any
from img_to_vid import concatenate_frames_to_video

BASE_DIR = Path(__file__).resolve().parent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--res", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    return parser.parse_args()

def setup_directories(session_dir: Path):
    raw_dir = session_dir / "raw_frames"
    seg_dir = session_dir / "segmented_frames"
    plots_dir = session_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, seg_dir, plots_dir

def extract_frames_at_res(video_path: Path, output_dir: Path, res: Tuple[int, int]):
    w, h = res
    cmd = (f"ffmpeg -i {video_path} -vf 'scale={w}:{h},setsar=1:1' -q:v 2 "
           f"-start_number 0 {output_dir}/frame_%05d.jpg -y")
    subprocess.run(cmd, shell=True, check=True)

def apply_overlay(frame, mask, color, opacity):
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 1.0 - opacity, overlay, opacity, 0)

def generate_visual_benchmarks(csv_path: Path, plots_dir: Path, run_id: str):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(df['frame_id'], df['encoder_ms'], df['decoder_ms'], df['visual_ms'], df['write_ms'], 
                  labels=['Encoder (GPU)', 'Decoder (GPU)', 'Overlay (CPU)', 'Gravação em Disco'], alpha=0.8)
    plt.title(f'Análise de Latência Completa: {run_id}')
    plt.xlabel('ID do Frame')
    plt.ylabel('Tempo (ms)')
    plt.legend(loc='upper left')
    plt.savefig(plots_dir / "latency_breakdown.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.stackplot(df['frame_id'], df['visual_ms'], df['write_ms'], 
                  labels=['Overlay (CPU)', 'Gravação em Disco'], alpha=0.8, colors=['#81b1d3', '#fdb462'])
    plt.title(f'Gargalos de Pós-processamento: {run_id}')
    plt.xlabel('ID do Frame')
    plt.ylabel('Tempo (ms)')
    plt.legend(loc='upper left')
    plt.savefig(plots_dir / "latency_overhead_isolated.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='masks_count', y='decoder_ms', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f'Impacto das Detecções no Decoder: {run_id}')
    plt.xlabel('Número de Máscaras')
    plt.ylabel('Tempo de Decoder (ms)')
    plt.savefig(plots_dir / "masks_vs_decoder.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='frame_id', y='vram_mb', color='green', linewidth=2)
    plt.fill_between(df['frame_id'], df['vram_mb'], alpha=0.3, color='green')
    plt.ylim(5400, 5600)
    plt.title(f'Estabilidade de VRAM: {run_id}')
    plt.xlabel('ID do Frame')
    plt.ylabel('Uso de Memória (MB)')
    plt.savefig(plots_dir / "vram_stability.png")
    plt.close()

def run_task2_comprehensive():
    args = get_args()
    log_buffer = []

    def log_print(msg):
        print(msg)
        log_buffer.append(str(msg))

    config_path = BASE_DIR / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    res_parts = [int(x) for x in args.res.split(',')]
    res = (res_parts[1], res_parts[0])
    height_val = res[1]
    res_str = f"{res[0]}x{res[1]}"
    
    video_name = args.video
    video_path = BASE_DIR / "materials" / "RayBanRatio" / "1x" / video_name
    video_stem = Path(video_name).stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_prompt_map = {
        "windows": {"car window": {"color": (0, 140, 255), "label": "Janela"}},
        "tires": {"tires": {"color": (255, 0, 0), "label": "Pneu"}},
        "bumpers": {"bumpers": {"color": (65, 169, 76), "label": "Para-choque"}}
    }
    
    selected_classes = args.classes.split(',')
    prompt_config = {}
    for sc in selected_classes:
        if sc in full_prompt_map:
            prompt_config.update(full_prompt_map[sc])
    
    incidence_counts = {p_cfg["label"]: 0 for p_cfg in prompt_config.values()}
    
    num_classes = len(selected_classes)
    run_id = f"{height_val}p_{num_classes}cl_{video_stem}_{run_timestamp}"
    session_dir = BASE_DIR / "proc_img_out" / run_id
    raw_frames_dir, segmented_dir, plots_dir = setup_directories(session_dir)
    
    extract_frames_at_res(video_path, raw_frames_dir, res)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model().to(device).eval()
    processor = Sam3Processor(model)
    
    log_print(f"Warmup iniciado para {res_str}")
    dummy_img = Image.fromarray(np.zeros((res[1], res[0], 3), dtype=np.uint8))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        d_state = processor.set_image(dummy_img)
        processor.set_text_prompt(state=d_state, prompt="warmup")
    if device == "cuda":
        torch.cuda.synchronize()
    
    overall_start = time.time()
    mask_opacity = 0.4
    conf_threshold = config['model_params']['confidence_threshold']
    frame_paths = sorted(list(raw_frames_dir.glob("*.jpg")))
    
    benchmark_data = []

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for idx, path in enumerate(frame_paths):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            pil_image = Image.open(path).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            t_start_enc = time.time()
            state = processor.set_image(pil_image)
            enc_ms = (time.time() - t_start_enc) * 1000
            
            masks_in_frame = 0
            total_dec_ms = 0
            total_vis_ms = 0

            for prompt_text, p_cfg in prompt_config.items():
                t_dec_sub = time.time()
                output = processor.set_text_prompt(state=state, prompt=prompt_text)
                total_dec_ms += (time.time() - t_dec_sub) * 1000
                
                t_vis_sub = time.time()
                masks, scores, boxes = output["masks"], output["scores"], output["boxes"]
                if masks is not None:
                    for i, m_tensor in enumerate(masks):
                        if scores[i].item() >= conf_threshold:
                            masks_in_frame += 1
                            incidence_counts[p_cfg["label"]] += 1
                            m_np = m_tensor.cpu().numpy().squeeze()
                            frame_bgr = apply_overlay(frame_bgr, m_np, p_cfg["color"], mask_opacity)
                            if boxes is not None:
                                box = boxes[i].cpu().numpy().astype(int)
                                cv2.putText(frame_bgr, p_cfg["label"], (box[0], box[1] - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_cfg["color"], 2)
                total_vis_ms += (time.time() - t_vis_sub) * 1000
            
            t_start_write = time.time()
            out_filename = f"segmented_{idx:05d}.jpg"
            out_path = segmented_dir / out_filename
            cv2.imwrite(str(out_path), frame_bgr)
            write_ms = (time.time() - t_start_write) * 1000
            
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == "cuda" else 0
            
            benchmark_data.append({
                "frame_id": idx, "resolution": res_str,
                "encoder_ms": round(enc_ms, 2), "decoder_ms": round(total_dec_ms, 2),
                "visual_ms": round(total_vis_ms, 2), "write_ms": round(write_ms, 2),
                "vram_mb": round(vram_mb, 2), "masks_count": masks_in_frame
            })
            log_print(f"[{out_filename}] Enc: {enc_ms:>6.2f}ms | Dec: {total_dec_ms:>6.2f}ms | Vis: {total_vis_ms:>6.2f}ms | Write: {write_ms:>6.2f}ms | VRAM: {vram_mb:>6.1f}MB | Masks: {masks_in_frame}")

    csv_path = session_dir / "benchmark_results.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_data[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_data)

    generate_visual_benchmarks(csv_path, plots_dir, run_id)
    
    total_duration = time.time() - overall_start
    enc_l = [d['encoder_ms'] for d in benchmark_data]
    dec_l = [d['decoder_ms'] for d in benchmark_data]
    vis_l = [d['visual_ms'] for d in benchmark_data]
    write_l = [d['write_ms'] for d in benchmark_data]

    log_print("\n" + "="*90)
    log_print(f"Tempo Total: {total_duration / 60:.2f} minutos")
    log_print(f"Média Encoder: {statistics.mean(enc_l):.2f}ms (±{statistics.stdev(enc_l):.2f}ms)")
    log_print(f"Média Decoder: {statistics.mean(dec_l):.2f}ms (±{statistics.stdev(dec_l):.2f}ms)")
    log_print(f"Média Visual:  {statistics.mean(vis_l):.2f}ms (±{statistics.stdev(vis_l):.2f}ms)")
    log_print(f"Média Escrita: {statistics.mean(write_l):.2f}ms (±{statistics.stdev(write_l):.2f}ms)")
    log_print(f"Total de Mascaras Geradas: {sum(incidence_counts.values())}")
    for label, count in incidence_counts.items():
        log_print(f"Incidencia {label}: {count} deteccoes")
    log_print("="*90)

    with open(session_dir / "session_log.txt", "w") as f:
        f.write("\n".join(log_buffer))

if __name__ == "__main__":
    run_task2_comprehensive()