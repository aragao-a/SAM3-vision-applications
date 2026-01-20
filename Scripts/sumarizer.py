import pandas as pd
from pathlib import Path

def calculate_total_masks():
    base_dir = Path(__file__).parent.resolve() / "proc_img_out"
    output_path = base_dir / "contagem_total_mascaras.txt"
    
    report_lines = []
    
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    for subdir in subdirs:
        csv_path = subdir / "benchmark_results.csv"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                total_masks = int(df['masks_count'].sum())
                
                line = f"SESSAO: {subdir.name}\nTOTAL DE MASCARAS: {total_masks}"
                report_lines.append(line + "\n" + "-"*50)
            except Exception as e:
                print(f"Erro ao processar {subdir.name}: {e}")
                
    if report_lines:
        output_path.write_text("\n".join(report_lines))
    else:
        print("Nada encontrado")

if __name__ == "__main__":
    calculate_total_masks()