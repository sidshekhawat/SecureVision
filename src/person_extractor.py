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
extracted_persons/*.jpg
"""

from ultralytics import YOLO
import cv2
import os
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Create folder
OUTPUT_DIR = ROOT / "extracted_persons"
REPORTS = ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS.mkdir(
    parents=True,
    exist_ok=True
)

# Load model
MODEL_PATH = ROOT / "models" / "yolov8n.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

# Open video
VIDEO_PATH = ROOT / "input_videos" / "4.avi"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError(
        "Invalid FPS detected."
    )

frame_count = 0

existing_files = list(
    OUTPUT_DIR.glob("person_*.jpg")
)

image_count = len(existing_files)

csv_path = REPORTS / "timestamps.csv"

csv_file = open(
    csv_path,
    "w",
    newline=""
)
writer = csv.writer(csv_file)
writer.writerow(["Image", "Timestamp"])

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

            save_path = OUTPUT_DIR / filename

            cv2.imwrite(
                str(save_path),
                person_crop
            )

            timestamp = frame_count / fps

            writer.writerow([
                filename,
                f"{timestamp:.2f}"
            ])

            image_count += 1

    cv2.imshow("ATM Person Extractor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"Saved {image_count} images")