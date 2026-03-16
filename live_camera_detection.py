import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()