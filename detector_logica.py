import cv2
from ultralytics import YOLO

import config 

class DetectorFrutas:
    def __init__(self):
        
        self.model = YOLO(config.MODEL_PATH)
        print(f"Modelo cargado desde: {config.MODEL_PATH}")

    def procesar_frame(self, frame):
        """
        Recibe un frame (imagen) y devuelve el frame con las cajas dibujadas
        y la lista de detecciones.
        """
        # 1. Preprocesamiento (usamos el tamaño definido en config)
        frame_resized = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        # 2. Inferencia (Predicción)
        resultados = self.model.predict(
            source=frame_resized,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMAGE_SIZE,
            verbose=False 
        )
        
        # 3. Postprocesamiento (Dibujar resultados)
        # El método plot() de YOLO devuelve la imagen con las cajas ya dibujadas
        frame_con_cajas = resultados[0].plot()
        
        # 4. Extraer información (útil para lógica de negocio)
        detecciones = resultados[0].boxes
        
        return frame_con_cajas, detecciones

    def contar_frutas(self, detecciones):
        """Ejemplo de función extra: contar cuántas frutas detectó"""
        return len(detecciones)