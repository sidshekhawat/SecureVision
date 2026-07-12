from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8s.pt")

# Open video
cap = cv2.VideoCapture("input_videos/1.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()