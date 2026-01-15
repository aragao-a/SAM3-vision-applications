import subprocess
import time
import sys
from pathlib import Path

def run_master_benchmark():
    script_dir = Path(__file__).parent.resolve()
    bench_script = script_dir / "bench_color_extractor.py"

    videos = ["GT7-Easy3-30.mp4", "GT7-Easy2-30.mp4", "GT7-Easy1-30.mp4"]
    resolutions = ["3572,2680", "894,670"]
    class_sets = [
        "tires,windows,bumpers",
        "tires,windows",
        "tires"
    ]

    total_runs = len(videos) * len(resolutions) * len(class_sets)
    current_run = 1

    print(f"Iniciando ({total_runs} execucoes no total")
    start_time_all = time.time()

    for video in videos:
        for res in resolutions:
            for classes in class_sets:
                print(f"\nExecução {current_run}/{total_runs}")
                print(f"> Video: {video} | Res: {res} | Classes: {classes}")
                
                cmd = [
                    sys.executable, 
                    str(bench_script),
                    "--video", video,
                    "--res", res,
                    "--classes", classes
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Erro na execução {current_run}: {e}")
                
                current_run += 1

    total_duration = (time.time() - start_time_all) / 60
    print("\n" + "="*50)
    print(f"Master Benchmark feito em {total_duration:.2f} minutos")
    print("="*50)

if __name__ == "__main__":
    run_master_benchmark()