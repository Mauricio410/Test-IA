# Cam video

import torch
import cv2

# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

def detector():
    # Capturar el video
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Realizar la predicción
        prediction = model(frame)

        # Convertir las predicciones a un DataFrame
        df = prediction.pandas().xyxy[0]

        # Filtrar las predicciones por confianza
        df = df[df["confidence"] > 0.5]

        # Dibujar los bounding boxes en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            # frame -> (xmin, ymin); (xmax, ymax)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Clase de cada bbox
            label = f"{df.iloc[i]['name']} {df.iloc[i]['confidence']:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar el frame using cv2_imshow(), no se puede usar cv2.imshow() porque no es compatible con GC
        cv2.imshow("IA Video", frame)

        # Salir si se presiona la tecla 'Esc'
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector()