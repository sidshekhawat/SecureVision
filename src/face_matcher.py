"""
face_matcher.py

Purpose:
Matches extracted faces against watchlist database.

Workflow:
1. Loads watchlist images
2. Extracts face embeddings
3. Compares faces
4. Reports matches

Technology:
DeepFace
"""
from deepface import DeepFace
import os
import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KNOWN_FACES = ROOT / "watchlist"
EXTRACTED_FACES = ROOT / "extracted_faces"
REPORTS = ROOT / "reports"

csv_file = REPORTS / "timestamp.csv"

REPORTS.mkdir(exist_ok=True)
if not csv_file.exists():
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "KnownFace",
            "DetectedFace",
            "Distance"
        ])

watchlist_images = [
    f for f in os.listdir(KNOWN_FACES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

extracted_images = [
    f for f in os.listdir(EXTRACTED_FACES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for known in watchlist_images:

    known_path = KNOWN_FACES / known

    print("\n" + "=" * 50)
    print(f"Searching for matches of: {known}")
    print("=" * 50)

    found = False

    for face in extracted_images:

        face_path = EXTRACTED_FACES / face

        try:
            result = DeepFace.verify(
                img1_path=str(known_path),
                img2_path=str(face_path),
                model_name="VGG-Face",
                enforce_detection=False,
                threshold=0.55
            )

            print(f"\nChecking {face}")
            print("Verified:", result["verified"])
            print("Distance:", result["distance"])
            print("Threshold:", result["threshold"])

            if result["verified"]:
                print(f"MATCH FOUND -> {face}")
                found = True
                distance = result["distance"]
                confidence = round((1 - distance) * 100, 2)
                print(f"Confidence: {confidence}%")
                with open(csv_file, "a", newline="") as file:
                     writer = csv.writer(file)

                     writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            known,
                            face,
                            distance
            ])

        except Exception as e:
            print(f"Error with {face}: {e}")

    if not found:
        print("No matches found.")