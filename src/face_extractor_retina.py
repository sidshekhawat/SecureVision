from retinaface import RetinaFace
import cv2
import os

os.makedirs("extracted_faces", exist_ok=True)

image_count = 0

for filename in os.listdir("extracted_persons"):

    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join("extracted_persons", filename)

    image = cv2.imread(image_path)

    try:

        faces = RetinaFace.detect_faces(image)

        if isinstance(faces, dict):

            first_face = list(faces.values())[0]

            x1, y1, x2, y2 = first_face["facial_area"]

            face_crop = image[y1:y2, x1:x2]

            if face_crop.size > 0:

                output_name = f"face_{image_count:04d}.jpg"

                cv2.imwrite(
                    os.path.join("extracted_faces", output_name),
                    face_crop
                )

                image_count += 1

    except Exception:
        pass

print(f"Saved {image_count} faces")