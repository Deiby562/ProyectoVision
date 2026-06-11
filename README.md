# FruitQual: Sistema de Inspección Automatizada de Frutas

FruitQual es una solución de visión artificial orientada a la inspección de calidad y conteo de frutas. Utiliza el modelo **YOLO11** (vía `ultralytics`) para realizar inferencias en tiempo real, permitiendo alternar entre captura por cámara web y análisis de archivos locales (imágenes/videos).

## 🚀 Características Principales
- **Pipeline Híbrido:** Procesa flujos de video en vivo o archivos estáticos con la misma lógica de inferencia.
- **Control de Reproducción:** Incluye navegación en archivos de video (adelantar/atrasar 5 segundos).
- **Conteo Inteligente:** Algoritmo modular para conteo de objetos detectados con persistencia en logs.
- **Interfaz GUI:** Desarrollo en `Tkinter` con diseño profesional en colores oscuros para reducción de fatiga visual.
- **Configuración Centralizada:** Parámetros como umbrales de confianza, IoU y resolución ajustables vía `config.py`.

## 🛠 Estructura del Proyecto
```text
ProyectoVision/
├── Modelos/             # Carpeta contenedora de los pesos (.pt)
├── detector_logica.py   # Núcleo de inferencia y postprocesamiento
├── GUI.py               # Interfaz gráfica y gestión de eventos
├── utils.py             # Funciones auxiliares (dibujo, logging)
├── config.py            # Parámetros de configuración del modelo
└── FruitQual.ipynb      # Notebook para experimentación y batch processing