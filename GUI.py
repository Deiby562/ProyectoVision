import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Módulos para lógica separada
from detector_logica import DetectorFrutas
import utils
import config

class FruitQualApp:
    def __init__(self, window):
        self.window = window
        self.window.title("FruitQual - Sistema de Inspección Automatizada")
        self.window.geometry("1100x650")
        self.window.configure(bg="#2c3e50")

        # Inicialización del detector modular
        self.detector = DetectorFrutas()
        
        self.cap = None
        self.camara_activa = False
        self.es_camara = False
      
        self.crear_componentes_gui()
        
        # Iniciar cámara automáticamente al abrir
        self.window.after(500, self.conmutar_camara)

    def crear_componentes_gui(self):
        """Estructura visual original mantenida"""
        titulo = tk.Label(
            self.window, 
            text="FruitQual: Inspección de Calidad de Frutas (YOLO11)", 
            font=("Arial", 18, "bold"), fg="#ecf0f1", bg="#2c3e50", pady=10
        )
        titulo.pack()

        panel_botones = ttk.Frame(self.window, padding=10)
        panel_botones.pack(fill="x", padx=20, pady=5)

        btn_cargar = tk.Button(
            panel_botones, text=" Cargar Archivo Local", font=("Arial", 11, "bold"),
            bg="#2ecc71", fg="white", width=22, command=self.procesar_archivo_local
        )
        btn_cargar.pack(side="left", padx=20)

        self.btn_camara = tk.Button(
            panel_botones, text=" Detener Cámara", font=("Arial", 11, "bold"),
            bg="#e67e22", fg="white", width=22, command=self.conmutar_camara
        )
        self.btn_camara.pack(side="left", padx=20)

        btn_salir = tk.Button(
            panel_botones, text=" Salir", font=("Arial", 11, "bold"),
            bg="#e74c3c", fg="white", width=12, command=self.cerrar_aplicacion
        )
        btn_salir.pack(side="right", padx=20)

        self.panel_pantallas = tk.Frame(self.window, bg="#2c3e50")
        self.panel_pantallas.pack(fill="both", expand=True, padx=20, pady=10)

        self.marco_izq = ttk.LabelFrame(self.panel_pantallas, text=" ETAPA 1 y 2: Entrada / Preprocesamiento ")
        self.marco_izq.pack(side="left", fill="both", expand=True, padx=10, pady=5)
        self.lbl_original = tk.Label(self.marco_izq, bg="#34495e")
        self.lbl_original.pack(fill="both", expand=True, padx=5, pady=5)

        self.marco_der = ttk.LabelFrame(self.panel_pantallas, text=" ETAPA 4 y 5: Reconocimiento / Postprocesamiento ")
        self.marco_der.pack(side="right", fill="both", expand=True, padx=10, pady=5)
        self.lbl_resultado = tk.Label(self.marco_der, bg="#34495e")
        self.lbl_resultado.pack(fill="both", expand=True, padx=5, pady=5)

        self.lbl_estado = tk.Label(self.window, text="Iniciando sistema...", bd=1, relief="sunken", anchor="w", font=("Arial", 10), bg="#bdc3c7", fg="#2c3e50")
        self.lbl_estado.pack(fill="x", side="bottom")
        
        # Dentro de crear_componentes_gui, después del botón de cámara:
        btn_atras = tk.Button(panel_botones, text="-5s", command=lambda: self.saltar_video(-5))
        btn_atras.pack(side="left", padx=5)

        btn_adelante = tk.Button(panel_botones, text="+5s", command=lambda: self.saltar_video(5))
        btn_adelante.pack(side="left", padx=5)

    def ejecutar_pipeline(self, frame_bgr):
        if frame_bgr is None: return
        
        # Inferencia delegada
        frame_anotado, detecciones = self.detector.procesar_frame(frame_bgr)
        cantidad = self.detector.contar_frutas(detecciones)
        
        # Utils para dibujo y logs
        frame_anotado = utils.dibujar_contador(frame_anotado, cantidad)
        utils.guardar_log(cantidad)
        
        self.lbl_estado.config(text=f"Pipeline activo | Elementos detectados: {cantidad}")
        self.mostrar_en_label(self.lbl_original, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.mostrar_en_label(self.lbl_resultado, cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB))

    def procesar_archivo_local(self):
        """Manejador unificado para cargar imágenes o videos."""
        if self.camara_activa:
            self.conmutar_camara() # Apagar cámara si está encendida
            
        ruta_archivo = filedialog.askopenfilename(
            filetypes=[("Archivos Multimedia", "*.jpg *.jpeg *.png *.avi *.mp4")]
        )
        
        if not ruta_archivo: return

        # Si es imagen
        if ruta_archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_nativa = cv2.imread(ruta_archivo)
            self.ejecutar_pipeline(img_nativa)
            
        # Si es video
        elif ruta_archivo.lower().endswith(('.avi', '.mp4')):
            self.cap = cv2.VideoCapture(ruta_archivo)
            if self.cap.isOpened():
                self.camara_activa = True
                self.es_camara = False
                self.btn_camara.config(text="Detener Video", bg="#e67e22")
                self.actualizar_video_en_vivo()
            else:
                messagebox.showerror("Error", "No se pudo abrir el archivo de video.")

    def conmutar_camara(self):
        if not self.camara_activa:
            # Lógica robusta de apertura
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
                if cap.isOpened():
                    for _ in range(10): cap.read() # Warmup
                    self.cap = cap
                    self.camara_activa = True
                    self.btn_camara.config(text="Detener Cámara", bg="#e67e22")
                    self.actualizar_video_en_vivo()
                    return
                cap.release()
            messagebox.showerror("Error", "No se detectó cámara web.")
        else:
            self.camara_activa = False
            if self.cap: self.cap.release()
            self.es_camara = False
            self.btn_camara.config(text=" Activar Cámara Web", bg="#3498db")
            self.lbl_estado.config(text="Cámara web desconectada.")

    def actualizar_video_en_vivo(self):
        if self.camara_activa and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Solo volteamos si es cámara web REAL
                if self.es_camara:
                    frame = cv2.flip(frame, 1)
                
                self.ejecutar_pipeline(frame)
            self.window.after(15, self.actualizar_video_en_vivo)

    def saltar_video(self, segundos):
        """Salta una cantidad de segundos en el video actual"""
        if self.cap and self.cap.isOpened():
            # Obtener FPS para calcular el salto en frames
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30 # Valor por defecto si no se detectan FPS
            
            # Calcular frame actual y nuevo
            frame_actual = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            salto_frames = int(segundos * fps)
            nuevo_frame = max(0, frame_actual + salto_frames)
            
            # Aplicar el salto
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, nuevo_frame)
            
            # Forzar una actualización inmediata
            ret, frame = self.cap.read()
            if ret:
                self.ejecutar_pipeline(frame)

    def mostrar_en_label(self, label_target, img_rgb):
        img_pil = Image.fromarray(img_rgb).resize((480, 420), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_target.img_tk = img_tk
        label_target.config(image=img_tk)

    def cerrar_aplicacion(self):
        self.camara_activa = False
        if self.cap: self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitQualApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar_aplicacion)
    root.mainloop()