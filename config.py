# config.py
""""
Configuración centralizada para el proyecto de detección de frutas.
"""

# --- Rutas ---
MODEL_PATH = "Modelos/best3.pt"
# Puedes añadir rutas a carpetas de log o exportación
OUTPUT_FOLDER = "Resultados/"

# --- Parámetros de Inferencia (YOLO) ---
CONFIDENCE_THRESHOLD = 0.6  # Ajusta qué tan "seguro" debe estar el modelo
IOU_THRESHOLD = 0.45        # Evita cajas duplicadas sobre el mismo objeto
IMAGE_SIZE = 512            # Tamaño al que se redimensionan las imágenes

# --- Configuración de Cámara ---
CAMERA_INDEX = 0            # 0 es la cámara por defecto
FPS_LIMIT = 30              # Controla la fluidez del video

# --- Etiquetas (Opcional, ayuda a la UI) ---
LABELS = ["Manzana", "Naranja", "Plátano"]