import cv2
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"

def get_latest_inference_dir():
    if not RUNS_DIR.exists():
        print(f"Erro: Pasta {RUNS_DIR} nao encontrada.")
        return None
    
    # Filtra apenas pastas que comecam com 'batch_'
    batches = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    
    if not batches:
        print("Nenhuma pasta de execucao encontrada em 'runs/'.")
        return None
        
    # Ordena por nome (timestamp) para pegar a mais recente
    latest_run = sorted(batches, reverse=True)[0]
    inference_dir = latest_run / "inference"
    
    if not inference_dir.exists():
        print(f"Erro: Pasta de inferencia nao encontrada em {latest_run}")
        return None
        
    return inference_dir

def run_viewer():
    inf_dir = get_latest_inference_dir()
    if not inf_dir:
        sys.exit(1)
        
    print(f"-> Exibindo resultados de: {inf_dir}")
    
    images = sorted(list(inf_dir.glob("*.jpg")))
    
    if not images:
        print("Nenhuma imagem encontrada na pasta de inferencia.")
        sys.exit(1)

    window_name = "SAM3 Result Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            continue
            
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Visualizacao encerrada.")

if __name__ == "__main__":
    run_viewer()