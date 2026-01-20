import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = [
    {"vid": "Easy1", "res": "3572p", "cls": "1cl", "enc": 76.78, "dec": 762.29, "vis": 144.50, "masks": 1249},
    {"vid": "Easy1", "res": "3572p", "cls": "2cl", "enc": 72.00, "dec": 772.16, "vis": 233.16, "masks": 2002},
    {"vid": "Easy1", "res": "3572p", "cls": "3cl", "enc": 78.89, "dec": 1299.27, "vis": 291.10, "masks": 2190},
    {"vid": "Easy1", "res": "894p", "cls": "1cl", "enc": 30.78, "dec": 504.41, "vis": 7.76, "masks": 1249},
    {"vid": "Easy1", "res": "894p", "cls": "2cl", "enc": 30.00, "dec": 610.69, "vis": 12.40, "masks": 1972},
    {"vid": "Easy1", "res": "894p", "cls": "3cl", "enc": 30.00, "dec": 800.00, "vis": 14.86, "masks": 2090},
    {"vid": "Easy2", "res": "3572p", "cls": "1cl", "enc": 62.14, "dec": 2092.31, "vis": 262.28, "masks": 2496},
    {"vid": "Easy2", "res": "3572p", "cls": "2cl", "enc": 64.67, "dec": 1965.99, "vis": 523.94, "masks": 4961},
    {"vid": "Easy2", "res": "3572p", "cls": "3cl", "enc": 63.58, "dec": 1991.68, "vis": 633.14, "masks": 5846},
    {"vid": "Easy2", "res": "894p", "cls": "1cl", "enc": 47.93, "dec": 781.21, "vis": 19.61, "masks": 2453},
    {"vid": "Easy2", "res": "894p", "cls": "2cl", "enc": 45.46, "dec": 813.99, "vis": 36.90, "masks": 4778},
    {"vid": "Easy2", "res": "894p", "cls": "3cl", "enc": 44.04, "dec": 998.53, "vis": 39.58, "masks": 5445},
    {"vid": "Easy3", "res": "3572p", "cls": "1cl", "enc": 62.07, "dec": 1431.12, "vis": 147.80, "masks": 2031},
    {"vid": "Easy3", "res": "3572p", "cls": "2cl", "enc": 66.94, "dec": 1648.58, "vis": 273.50, "masks": 3917},
    {"vid": "Easy3", "res": "3572p", "cls": "3cl", "enc": 59.46, "dec": 1792.07, "vis": 315.88, "masks": 4095},
    {"vid": "Easy3", "res": "894p", "cls": "1cl", "enc": 31.89, "dec": 504.02, "vis": 8.91, "masks": 2025},
    {"vid": "Easy3", "res": "894p", "cls": "2cl", "enc": 36.82, "dec": 1180.55, "vis": 18.32, "masks": 3878},
    {"vid": "Easy3", "res": "894p", "cls": "3cl", "enc": 32.71, "dec": 1153.04, "vis": 18.11, "masks": 3981}
]

df = pd.DataFrame(data)
sns.set_theme(style="whitegrid")

def plot_encoder_vs_resolution():
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='res', y='dec', hue='vid', errorbar=None)
    plt.title('Impacto da Resolucao na Latencia do Decoder (GPU)')
    plt.ylabel('Tempo Medio (ms)')
    plt.xlabel('Resolucao')
    plt.savefig('plot_dec_resolution.png')
    plt.close()

def plot_decoder_by_classes():
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df[df['res'] == '3572p'], x='vid', y='dec', hue='cls')
    plt.title('Escalabilidade do Decoder por Numero de Classes (3572p)')
    plt.ylabel('Tempo Acumulado do Decoder (ms)')
    plt.xlabel('Cenario de Video')
    plt.savefig('plot_decoder_classes.png')
    plt.close()

def plot_mask_retention():
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='vid', y='masks', hue='res')
    plt.title('Consistencia de Deteccao: Alta vs Baixa Resolucao')
    plt.ylabel('Total de Mascaras Detectadas')
    plt.xlabel('Cenario de Video')
    plt.savefig('plot_mask_retention.png')
    plt.close()

def plot_bottleneck_analysis():
    heavy_case = df[(df['res'] == '3572p') & (df['vid'] == 'Easy2') & (df['cls'] == '3cl')].iloc[0]
    metrics = ['Encoder (GPU)', 'Decoder (GPU)', 'Visual (CPU)']
    values = [heavy_case['enc'], heavy_case['dec'], heavy_case['vis']]
    
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=metrics, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Distribuicao de Carga no Cenario Critico (3572p, Easy2, 3cl)')
    plt.savefig('plot_bottleneck_pie.png')
    plt.close()

if __name__ == "__main__":
    plot_encoder_vs_resolution()
    plot_decoder_by_classes()
    plot_mask_retention()
    plot_bottleneck_analysis()