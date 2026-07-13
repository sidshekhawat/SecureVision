"""
face_extractor_retina.py

Purpose:
Detects and extracts faces from person images using RetinaFace.

Workflow:
1. Reads cropped person images
2. Detects faces using RetinaFace
3. Crops face region
4. Saves extracted faces into extracted_faces/

Input:
extracted_persons/*.jpg

Output:
extracted_faces/*.jpg

Technology:
RetinaFace

Why RetinaFace?
Provides more accurate face detection than Haar Cascades,
especially for CCTV and ATM surveillance footage.
"""
from ultralytics import YOLO
from retinaface import RetinaFace
import cv2
import os
import csv

# Create folders
os.makedirs("extracted_faces", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("input_videos/1.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0

# Continue numbering from existing files
existing_files = [
    f for f in os.listdir("extracted_faces")
    if f.startswith("face_") and f.endswith(".jpg")
]

image_count = len(existing_files)

# CSV report
csv_file = open("reports/timestamps.csv", "w", newline="")
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

    # Find largest person
    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls == 0:  # person

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

            try:

                faces = RetinaFace.detect_faces(person_crop)

                if isinstance(faces, dict):

                    first_face = list(faces.values())[0]

                    fx1, fy1, fx2, fy2 = first_face["facial_area"]

                    face_crop = person_crop[fy1:fy2, fx1:fx2]

                    if face_crop.size > 0:

                        filename = f"face_{image_count:04d}.jpg"

                        cv2.imwrite(
                            f"extracted_faces/{filename}",
                            face_crop
                        )

                        timestamp = frame_count / fps

                        writer.writerow([
                            filename,
                            f"{timestamp:.2f}"
                        ])

                        image_count += 1

            except Exception:
                pass

    cv2.imshow("ATM Face Extractor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"Saved {image_count} faces")