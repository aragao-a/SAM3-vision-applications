import cv2
import time
import yaml
from pathlib import Path

# Configuração de caminhos
BASE_DIR = Path(__file__).resolve().parent
MATERIALS_DIR = BASE_DIR / "materials" / "1080p"

def load_config():
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def start_live_simulation():
    config = load_config()
    sim_cfg = config['simulation']
    
    video_path = MATERIALS_DIR / sim_cfg['video_name']
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
        return

    # Obtendo metadados para simular o "tempo real" 
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps # Tempo de espera entre frames em segundos
    
    print(f"-> Simulando Playback: {sim_cfg['video_name']} ({fps} FPS)")
    print(f"-> Snapshot a cada {sim_cfg['frame_snap_interval']} frames [cite: 11]")

    frame_idx = 0
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # --- PARTE PARA MODIFICAÇÃO FUTURA (Task 2/Streaming) ---
        # Aqui simularemos o downscale/upscale e a qualidade de streaming [cite: 10, 29]
        # frame = cv2.resize(frame, tuple(sim_cfg['input_resolution'])) 
        # --------------------------------------------------------

        # Lógica de identificação de Snapshot [cite: 11]
        is_snapshot = (frame_idx % sim_cfg['frame_snap_interval'] == 0)
        
        # Feedback visual temporário para o Snapshot
        display_frame = frame.copy()
        if is_snapshot:
            # Desenha um indicador de "Captura" na tela
            cv2.circle(display_frame, (50, 50), 20, (0, 255, 0), -1)
            cv2.putText(display_frame, "SNAPSHOT", (80, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibe o feed "ao vivo" em uma janela [cite: 20]
        cv2.imshow("RayBan Meta - Live Simulation Feed", display_frame)

        # Controle de FPS: Garante que o vídeo não rode rápido demais 
        elapsed = time.time() - start_time
        sleep_time = max(1, int((frame_time - elapsed) * 1000))
        
        frame_idx += 1

        # Interromper com 'q'
        if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_simulation()