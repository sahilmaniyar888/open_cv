import cv2

# Load video (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise using threshold
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    burglar_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 1500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            burglar_detected = True

    if burglar_detected:
        cv2.putText(
            frame,
            "Burglar Detected!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Motion Mask", thresh)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
