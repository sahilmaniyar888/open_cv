import os
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
img1 = cv2.imread(os.path.join(script_dir, "cfd.png"))

resizing = cv2.resize(img1, (300, 300))
grey = cv2.cvtColor(resizing, cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(grey, (5, 5), 0)
edging = cv2.Canny(blurring, 50, 150)


cv2.imshow("Resized Image", resizing)
cv2.imshow("Grayscale Image", grey)
cv2.imshow("Blurred Image", blurring)
cv2.imshow("Edged Image", edging)
cv2.waitKey(0)