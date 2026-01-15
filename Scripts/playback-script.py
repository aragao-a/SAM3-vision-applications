import cv2
import time
import yaml
from pathlib import Path

# Configuração de caminhos
BASE_DIR = Path(__file__).resolve().parent
MATERIALS_DIR = BASE_DIR / "materials" / "1080p"
TARGET_RES = (1280, 720) # Downscale para 720p

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def start_live_simulation():
    config = load_config()
    sim_cfg = config['simulation']
    
    video_path = MATERIALS_DIR / sim_cfg['video_name']
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    
    print(f"-> Simulando Playback (720p): {sim_cfg['video_name']}")

    frame_idx = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # APLICAÇÃO DO DOWNSCALE AUTOMÁTICO
        frame = cv2.resize(frame, tuple(sim_cfg['output_resolution']), interpolation=cv2.INTER_AREA)

        is_snapshot = (frame_idx % sim_cfg['frame_snap_interval'] == 0)
        display_frame = frame.copy()
        
        if is_snapshot:
            # Lógica de salvar snapshot (Task 1 original)
            cv2.circle(display_frame, (40, 40), 15, (0, 255, 0), -1)
            cv2.putText(display_frame, "SNAP", (65, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live Feed - 720p View", display_frame)

        elapsed = time.time() - start_time
        sleep_time = max(1, int((frame_time - elapsed) * 1000))
        frame_idx += 1

        if cv2.waitKey(sleep_time) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_simulation()