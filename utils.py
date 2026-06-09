import cv2
import datetime

def dibujar_contador(frame, cantidad):
    """Dibuja un contador elegante en la esquina superior izquierda."""
    cv2.putText(frame, f"Frutas detectadas: {cantidad}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def guardar_log(frutas_detectadas):
    """Guarda un registro simple de la detección."""
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("historial.txt", "a") as f:
        f.write(f"{fecha} - Frutas: {frutas_detectadas}\n")