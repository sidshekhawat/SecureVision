"""
person_extractor.py

Purpose:
Detects persons in ATM/CCTV video using YOLOv8.

Workflow:
1. Reads video frames
2. Detects persons
3. Crops detected persons
4. Saves them into extracted_faces/

Input:
input_videos/*.mp4

Output:
extracted_faces/*.jpg
"""

from ultralytics import YOLO
import cv2
import os

# Create folder
os.makedirs("extracted_faces", exist_ok=True)

# Load model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("input_videos/1.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0

existing_files = [
    f for f in os.listdir("extracted_faces")
    if f.startswith("person_") and f.endswith(".jpg")
]

image_count = len(existing_files)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % 60 != 0:
        continue

    results = model(frame, verbose=False)

    largest_person = None
    largest_area = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])

        # class 0 = person
        if cls == 0:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                largest_person = (x1, y1, x2, y2)

    if largest_person:

        x1, y1, x2, y2 = largest_person

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Save every 2 seconds
        if frame_count % int(fps * 2) == 0:

            person_crop = frame[y1:y2, x1:x2]

            filename = f"person_{image_count:04d}.jpg"

            cv2.imwrite(
                f"extracted_faces/{filename}",
                person_crop
            )

            image_count += 1

    cv2.imshow("ATM Person Extractor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {image_count} images")