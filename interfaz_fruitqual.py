import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

class FruitQualApp:
    def __init__(self, window):
        self.window = window
        self.window.title("FruitQual - Sistema de Inspección Automatizada")
        self.window.geometry("1100x650")
        self.window.configure(bg="#2c3e50") # Fondo elegante oscuro

        # 1. CARGA DEL MODELO ENTRENADO (Etapa de Reconocimiento)
        self.ruta_modelo = "Modelos/best.pt"
        if os.path.exists(self.ruta_modelo):
            self.model = YOLO(self.ruta_modelo)
            print("¡Éxito! FruitQual (best.pt) cargado en la interfaz.")
        else:
            messagebox.showerror("Error", f"No se encontró el modelo en: {self.ruta_modelo}\nSe usará el modelo base por defecto.")
            self.model = YOLO("Modelos/yolo11n.pt")

        # Variables de control del pipeline
        self.cap = None
        self.camara_activa = False

        self.crear_componentes_gui()

    def crear_componentes_gui(self):
        """Estructura visual de la interfaz bajo criterios de usabilidad"""
        
        # Título Principal
        titulo = tk.Label(
            self.window, 
            text="FruitQual: Inspección de Calidad de Frutas (YOLO11)", 
            font=("Arial", 18, "bold"), fg="#ecf0f1", bg="#2c3e50", pady=10
        )
        titulo.pack()

        # --- PANEL DE BOTONES (Control de Adquisición) ---
        panel_botones = ttk.Frame(self.window, padding=10)
        panel_botones.pack(fill="x", padx=20, pady=5)

        btn_cargar = tk.Button(
            panel_botones, text=" Cargar Imagen Local", font=("Arial", 11, "bold"),
            bg="#2ecc71", fg="white", width=22, command=self.procesar_imagen_local
        )
        btn_cargar.pack(side="left", padx=20)

        self.btn_camara = tk.Button(
            panel_botones, text=" Activar Cámara Web", font=("Arial", 11, "bold"),
            bg="#3498db", fg="white", width=22, command=self.conmutar_camara
        )
        self.btn_camara.pack(side="left", padx=20)

        btn_salir = tk.Button(
            panel_botones, text=" Salir", font=("Arial", 11, "bold"),
            bg="#e74c3c", fg="white", width=12, command=self.cerrar_aplicacion
        )
        btn_salir.pack(side="right", padx=20)

        # --- PANEL DE VISUALIZACIÓN (Pipeline Outputs) ---
        self.panel_pantallas = tk.Frame(self.window, bg="#2c3e50")
        self.panel_pantallas.pack(fill="both", expand=True, padx=20, pady=10)

        # Contenedor Izquierdo: Entrada Nativa
        self.marco_izq = ttk.LabelFrame(self.panel_pantallas, text=" ETAPA 1 y 2: Entrada / Preprocesamiento ")
        self.marco_izq.pack(side="left", fill="both", expand=True, padx=10, pady=5)
        
        self.lbl_original = tk.Label(self.marco_izq, bg="#34495e")
        self.lbl_original.pack(fill="both", expand=True, padx=5, pady=5)

        # Contenedor Derecho: Salida del Pipeline Anotada
        self.marco_der = ttk.LabelFrame(self.panel_pantallas, text=" ETAPA 4 y 5: Reconocimiento / Postprocesamiento ")
        self.marco_der.pack(side="right", fill="both", expand=True, padx=10, pady=5)
        
        self.lbl_resultado = tk.Label(self.marco_der, bg="#34495e")
        self.lbl_resultado.pack(fill="both", expand=True, padx=5, pady=5)

        # Barra de Estado Inferior (Métricas operacionales)
        self.lbl_estado = tk.Label(self.window, text="Sistema listo. Cargue una fuente de datos para iniciar el pipeline.", bd=1, relief="sunken", anchor="w", font=("Arial", 10), bg="#bdc3c7", fg="#2c3e50")
        self.lbl_estado.pack(fill="x", side="bottom")

    # --- LÓGICA DEL PIPELINE DE VISIÓN ---

    def ejecutar_pipeline(self, frame_bgr):
        """Ejecuta de forma secuencial las fases del pipeline sobre el frame"""
        if frame_bgr is None:
            return
            
        # ETAPA 2: Preprocesamiento (Redimensionamiento explícito a la red)
        frame_preprocesado = cv2.resize(frame_bgr, (640, 640))
        
        # Clonamos para mostrar la entrada limpia en el panel izquierdo
        img_pre_rgb = cv2.cvtColor(frame_preprocesado, cv2.COLOR_BGR2RGB)
        
        # ETAPAS 3 y 4: Extracción de Características e Inferencia con YOLO11
        id_inicio = cv2.getTickCount()
        resultados = self.model(frame_preprocesado, verbose=False, conf=0.6)
        id_fin = cv2.getTickCount()
        
        # Calcular tiempo de ejecución de la inferencia (Métrica solicitada por la rúbrica)
        tiempo = (id_fin - id_inicio) / cv2.getTickFrequency() * 1000
        
        # ETAPA 5: Postprocesamiento y Visualización (Generación de Anotaciones)
        frame_anotado = resultados[0].plot()
        img_anotada_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
        
        # Conteo de frutas detectadas en la muestra
        detecciones = len(resultados[0].boxes)
        self.lbl_estado.config(text=f"Pipeline completado en {tiempo:.1f} ms | Elementos evaluados: {detecciones} frutas detectadas.")

        # Renderizar en la UI de Tkinter
        self.mostrar_en_label(self.lbl_original, img_pre_rgb)
        self.mostrar_en_label(self.lbl_resultado, img_anotada_rgb)

    def procesar_imagen_local(self):
        """Manejador para la carga de archivos estáticos (.jpg, .png)"""
        if self.camara_activa:
            self.conmutar_camara() # Apagar cámara si está encendida
            
        ruta_archivo = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if ruta_archivo:
            # ETAPA 1: Adquisición
            img_nativa = cv2.imread(ruta_archivo)
            self.ejecutar_pipeline(img_nativa)

    def conmutar_camara(self):
        """Enciende o apaga el flujo de video en tiempo real"""
        if not self.camara_activa:
            # Intentar abrir el dispositivo de captura por defecto
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error de Hardware", "No se detectó una cámara web activa.")
                return
            self.camara_activa = True
            self.btn_camara.config(text="Detener Cámara", bg="#e67e22")
            self.actualizar_video_en_vivo()
        else:
            self.camara_activa = False
            if self.cap:
                self.cap.release()
            self.btn_camara.config(text=" Activar Cámara Web", bg="#3498db")
            self.lbl_estado.config(text="Cámara web desconectada.")

    def actualizar_video_en_vivo(self):
        """Ciclo dinámico que refresca los frames de la cámara de forma asíncrona"""
        if self.camara_activa and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Pasar el cuadro capturado en vivo directamente al pipeline estructurado
                self.ejecutar_pipeline(frame)
            # Volver a llamar a la función tras 15 milisegundos (Asegura ~30 FPS en la GUI)
            self.window.after(15, self.actualizar_video_en_vivo)

    def mostrar_en_label(self, label_target, img_rgb):
        """Función auxiliar para adaptar matrices de imágenes a Widgets de Tkinter"""
        # Redimensionar dinámicamente para encajar de forma estética en los paneles del profesor
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((480, 420), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_target.img_tk = img_tk  # Prevenir recolección de basura de Python
        label_target.config(image=img_tk)

    def cerrar_aplicacion(self):
        self.camara_activa = False
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitQualApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar_aplicacion)
    root.mainloop()