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


# utils.py
class ContadorFrutas:
    def __init__(self, clases_modelo):
        # clases_modelo es un diccionario como: {0: 'manzana', 1: 'naranja'...}
        # Creamos los totales basados en los valores del modelo
        self.totales = {nombre: 0 for nombre in clases_modelo.values()}
        self.ids_contados = set()

    def incrementar(self, clase, track_id):
        # Validamos que la clase exista (seguridad extra)
        if clase not in self.totales:
            self.totales[clase] = 0
            
        if track_id is not None and track_id not in self.ids_contados:
            self.totales[clase] += 1
            self.ids_contados.add(track_id)