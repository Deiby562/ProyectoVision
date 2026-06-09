import cv2
import os
from ultralytics import YOLO


# Define rutas y nombres de forma centralizada
MODEL_PATH    = 'Modelos/best.pt'
VIDEO_SOURCE  = "tu_video.mp4" 
OUTPUT_FOLDER = "resultados_inferencia"
PROJECT_NAME  = "analisis_tiempo_real"

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─── GESTIÓN DE RECURSOS ──────────────────────────────────────────────────────
def procesar_video():
    # Cargar modelo una sola vez
    model = YOLO(MODEL_PATH)
    
    # Abrir video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {VIDEO_SOURCE}")
        return

    print(f"Iniciando procesamiento: {PROJECT_NAME}")

    try:
        # Usamos stream=True para mantener el uso de memoria bajo control
        resultados = model.predict(source=VIDEO_SOURCE, stream=True, conf=0.5)

        for i, r in enumerate(resultados):
            # Obtener el frame anotado
            frame_plot = r.plot()
            
            # Guardar frame procesado
            output_path = os.path.join(OUTPUT_FOLDER, f"frame_{i:05d}.jpg")
            cv2.imwrite(output_path, frame_plot)
            
            # Feedback visual en consola
            if i % 30 == 0:
                print(f"Procesando frame: {i}")

    finally:
        # Asegurar liberación de recursos incluso si ocurre un error
        cap.release()
        print("Recursos liberados correctamente.")

if __name__ == "__main__":
    procesar_video()