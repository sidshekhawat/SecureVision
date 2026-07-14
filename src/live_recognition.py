"""
live_recognition.py

Purpose:
Performs real-time surveillance and watchlist recognition.

Workflow:
1. Captures live video feed
2. Detects persons and faces
3. Extracts facial features
4. Compares against watchlist database
5. Generates alerts for matches
6. Displays recognition results in real time

Input:
Live camera feed or CCTV stream

Output:
- Watchlist match alerts
- Recognized person information
- Saved evidence images

Technologies:
YOLOv8
RetinaFace
DeepFace

Use Case:
Law enforcement and surveillance monitoring for
identifying persons of interest in real time.
"""
import cv2
from retinaface import RetinaFace
from deepface import DeepFace
import os
import numpy as np
from datetime import datetime
import time
import csv
import sqlite3
import smtplib
from email.message import EmailMessage

def send_email_alert(person, image_path):
    sender_email = "YOURMAIL@gmail.com"
    app_password = "your pass word here"

    msg = EmailMessage()
    msg["Subject"] = "🚨 SecureVision Alert"
    msg["From"] = sender_email
    msg["To"] = sender_email

    msg.set_content(
        f"""
Watchlist Match Detected

Person: {person}

Please check the attached evidence image.
"""
    )

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename=image_path.split("/")[-1]
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

WATCHLIST="watchlist"

watchlist_images = [
    f for f in os.listdir(WATCHLIST)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
watchlist_embeddings = []
print("Loading known faces...")

for image in watchlist_images:

    path = os.path.join(WATCHLIST, image)

    try:

        embedding = DeepFace.represent(
            img_path=path,
            model_name="VGG-Face",
            enforce_detection=False
        )[0]["embedding"]

        watchlist_embeddings.append(
            (image, np.array(embedding))
        )

        print(f"Loaded: {image}")

    except Exception as e:

        print(f"Error loading {image}: {e}")
frame_count = 0
last_name = "Unknown"
last_confidence = 0
logged_names = set()
conn = sqlite3.connect("reports/securevision.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    person TEXT,
    status TEXT,
    image_path TEXT
)
""")

conn.commit()
unknown_start_time = None
unknown_saved = False

while True:

    ret, frame = cap.read()
    frame_count +=1

    if not ret:
        break

    try:

        faces = RetinaFace.detect_faces(frame)

        if isinstance(faces, dict):

            for face in faces.values():

                x1, y1, x2, y2 = face["facial_area"]

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                temp_face = "temp_face.jpg"
                cv2.imwrite(temp_face, face_crop)
                if frame_count % 10 != 0:

                    if last_name == "Unknown":
                        box_color = (0, 255, 255) #Yellow    
                    else:
                        box_color = (0, 0, 255) #Red

                    cv2.rectangle(
                     frame,
                     (x1, y1),
                     (x2, y2),
                     box_color,
                     2
                     )

                    cv2.putText(
                        frame,
                        last_name,
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        box_color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{last_confidence}%",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    continue

                best_match = "Unknown"
                best_distance = 999

                for known, known_embedding in watchlist_embeddings:
                     try:

                        live_embedding = DeepFace.represent(
                            img_path=temp_face,
                            model_name="VGG-Face",
                            enforce_detection=False
                        )[0]["embedding"]

                        live_embedding = np.array(live_embedding)

                        distance = np.linalg.norm(
                            known_embedding - live_embedding
                        )

                        if distance < best_distance:
                            best_distance = distance
                            if distance < 1.00:
                                name = known.split("_")[0]
                                best_match = name
                                print("MATCH FOUND:", best_match)

                            print(best_distance)

                     except:
                      pass
                confidence = max(
                    0,
                    round((1 - best_distance) * 100, 2)
                )

                last_name = best_match
                last_confidence = confidence
                #if best_match == "Unknown":

                    #if unknown_start_time is None:
                        #unknown_start_time = time.time()

                    #elapsed = time.time() - unknown_start_time

                    #if elapsed > 5 and not unknown_saved:

                         #filename = datetime.now().strftime(
                            #"unknown_faces/unknown_%Y%m%d_%H%M%S.jpg"
                         #)

                         #cv2.imwrite(filename, frame)

                         #print(f"INTRUDER DETECTED -> {filename}")

                         #unknown_saved = True

                #else:

                     #unknown_start_time = None
                     #unknown_saved = False
                if best_match != "Unknown" and best_match not in logged_names:

                 current_time = datetime.now().strftime(
                        "%d-%m-%Y %I:%M:%S %p"
                 )
                 import os

                 os.makedirs("alerts", exist_ok=True)

                 filename = f"alerts/{best_match}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

                 cv2.imwrite(filename, frame)

                 send_email_alert(best_match, filename)


                 cursor.execute(
                         """
                         INSERT INTO logs
                         (timestamp, person, status, image_path)
                         VALUES (?, ?, ?, ?)
                         """,
                         (
                             current_time,
                             best_match,
                             "Watchlist Match",
                             filename
                         )
                    )

                 conn.commit()

                 logged_names.add(best_match)
                 
                 print(
                         f"WATCHLIST MATCH -> {best_match} at {current_time}"
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"{best_match}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"{confidence}%",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    except:
        pass

    cv2.imshow(
        "ATM Face Scanner v2.0",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows