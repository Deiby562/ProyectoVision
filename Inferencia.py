from ultralytics import YOLO
import cv2

# 1. Cargar tu modelo entrenado
model = YOLO('best.pt')

# 2. Cargar el video que grabaste
video_path = 'video_capturado.avi'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Aplicar detección (Inferencia)
    # conf=0.25 es el umbral que ajustamos antes
    results = model.predict(frame, conf=0.25)

    # 4. Dibujar resultados en el frame
    # .plot() devuelve el frame con las cajas y etiquetas dibujadas
    annotated_frame = results[0].plot()

    # 5. Mostrar
    cv2.imshow('Deteccion de Frutas', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()