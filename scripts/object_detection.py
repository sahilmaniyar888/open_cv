import os
import cv2
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(script_dir, "yolov8n.pt"))

image = cv2.imread(os.path.join(script_dir, "yolo.jpg"))

results = model(image)

annoted_image = results[0].plot()

cv2.imshow("YOLOv8 Person Detection", annoted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
