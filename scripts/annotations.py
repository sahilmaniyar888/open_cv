import numpy as np
import cv2

canvas = np.zeros((512, 512, 3), dtype="uint8")

cv2.line(canvas, (0, 0), (511, 511), (255, 0, 0), 5)
cv2.rectangle(canvas, (384, 0), (510, 128), (255 , 0, 0), -1)
cv2.putText(canvas, "addition to the box", (5, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()