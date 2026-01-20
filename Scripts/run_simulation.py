import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def run_system():

    #watcher = SCRIPT_DIR / "playback-script.py"
    watcher = SCRIPT_DIR / "playback-script.py"
    viewer = SCRIPT_DIR / "img_viewer.py"

    print("Iniciando Processo de Inferência (Headless)...")
    proc_inference = subprocess.Popen([sys.executable, str(watcher)])

    print("Iniciando Visualizador de Resultados...")
    proc_viewer = subprocess.Popen([sys.executable, str(viewer)])

    try:

        proc_inference.wait()
        proc_viewer.wait()
    except KeyboardInterrupt:
        print("Encerrando processos...")
        proc_inference.terminate()
        proc_viewer.terminate()

if __name__ == "__main__":
    run_system()