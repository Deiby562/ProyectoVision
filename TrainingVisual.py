import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# Configura la ruta a tu archivo results.csv
ruta_csv = 'runs/detect/train/results.csv'

def visualizar_en_tiempo_real():
    plt.ion()  # Modo interactivo
    fig, ax = plt.subplots(figsize=(10, 6))

    while True:
        if os.path.exists(ruta_csv):
            try:
                # Leer el archivo CSV
                df = pd.read_csv(ruta_csv)
                # Limpiar nombres de columnas (espacios en blanco)
                df.columns = df.columns.str.strip()

                ax.clear()
                # Graficar métricas clave: mAP50(B) y mAP50-95(B)
                ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (Box)', marker='o')
                ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95 (Box)', marker='o')
                
                ax.set_title('Progreso de Entrenamiento (mAP)')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Precisión')
                ax.legend()
                ax.grid(True)
                
                plt.draw()
                plt.pause(5) # Se actualiza cada 5 segundos
            except Exception as e:
                print(f"Esperando datos... {e}")
                time.sleep(5)
        else:
            print("Esperando a que se cree el archivo results.csv...")
            time.sleep(5)

if __name__ == "__main__":
    visualizar_en_tiempo_real()