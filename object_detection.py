import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model for person detection
model = YOLO("yolov8n.pt") 
 # Use 'yolov8n.pt' for a lightweight model  

image = cv2.imread("yolo.jpg")


results = model(image)

annoted_image = results[0].plot()

cv2.imshow("YOLOv8 Person Detection", annoted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





