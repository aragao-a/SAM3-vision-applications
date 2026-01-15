import torch
import time
import yaml
import statistics
import os
import sys
import ast
from pathlib import Path
from PIL import Image
from typing import cast, Tuple
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def resolve_resolution(config) -> Tuple[int, int]:
    sim = config['simulation']
    res_raw = sim['output_resolution']
    
    if isinstance(res_raw, str) and 'pixel7resolutions[' in res_raw:
        try:
            idx = int(res_raw.split('[')[1].split(']')[0])
            res_val = sim['pixel7resolutions'][idx]
            
            if isinstance(res_val, str):
                res_val = ast.literal_eval(res_val)
                
            return cast(Tuple[int, int], tuple(res_val))
        except Exception:
            return (1786, 1340)
            
    if isinstance(res_raw, (list, tuple)) and len(res_raw) == 2:
        return cast(Tuple[int, int], tuple(res_raw))
        
    return (1786, 1340)

def run_benchmark():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"ERRO: Arquivo config.yaml nao encontrado.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    res = resolve_resolution(config)
    folder_name = config['simulation']['video_name']
    frames_path = script_dir / "materials" / "RayBanRatio" / "1x" / "frames" / folder_name
    
    if not frames_path.exists():
        print(f"ERRO: Diretorio nao encontrado: {frames_path}")
        return

    images = sorted(list(frames_path.glob("*.jpg")))
    if not images:
        print(f"ERRO: Nenhuma imagem encontrada em {folder_name}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    latencies = []
    vrams = []
    prompts = ["car window", "tires", "bumpers"]
    
    print(f"-> Benchmarking: {len(images)} imagens | Res: {res}")
    total_start_time = time.time()

    for img_p in images:
        img_raw = Image.open(img_p).resize(res)
        print(f"DEBUG: {img_p.name} Input Size: {img_raw.size}")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        with torch.autocast(device, dtype=torch.bfloat16):
            state = processor.set_image(img_raw)
            for p in prompts:
                processor.set_text_prompt(state=state, prompt=p)
        
        latency_ms = (time.time() - start_time) * 1000
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        latencies.append(latency_ms)
        vrams.append(vram_mb)
        
        print(f"ID: {img_p.name} | Latencia: {latency_ms:.1f}ms | VRAM: {vram_mb:.1f}MB")

    total_duration = time.time() - total_start_time
    mean_lat = statistics.mean(latencies)
    std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
    mean_vram = statistics.mean(vrams)

    print("\n" + "="*70)
    print(f"RESULTADOS DO BENCHMARK - {folder_name}")
    print(f"Resolucao de Teste: {res}")
    print(f"Latencia Media: {mean_lat:.2f}ms")
    print(f"Desvio Padrao: {std_lat:.2f}ms")
    print(f"VRAM Media: {mean_vram:.2f}MB")
    print(f"Tempo Total: {total_duration:.2f}s")
    print("-" * 70)
    print(f"{len(images)} imagens processadas na resolucao {res}, na pasta {folder_name}, usando uma GPU RTX 1000 Ada laptop,")
    print(f"com 2560 CUDA cores. O tempo de inferencia medio de {folder_name}")
    print(f"foi de {mean_lat:.2f}ms, com uso medio de VRAM {mean_vram:.2f}MB")
    print(f"e tempo de processamento total de {total_duration:.2f}s.")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()