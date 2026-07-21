import numpy as np
import cv2
from pathlib import Path
from retinaface import RetinaFace

print("Starting fusion...")

def align_face(image):

    faces = RetinaFace.detect_faces(image)

    if not isinstance(faces, dict):
        return None

    face = list(faces.values())[0]

    left_eye = face["landmarks"]["left_eye"]
    right_eye = face["landmarks"]["right_eye"]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    center = (
        image.shape[1] // 2,
        image.shape[0] // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    aligned = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0])
    )

    return aligned

def sharpness_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()
folder = Path("extracted_faces")

faces = []
scores = []

for img_path in folder.glob("*.jpg"):

    img = cv2.imread(str(img_path))

    aligned = align_face(img)

    if aligned is None:
        continue

    aligned = cv2.resize(
        aligned,
        (512, 512)
    )

    score = sharpness_score(aligned)

    faces.append(aligned)
    scores.append(score)

if len(faces) == 0:
    raise ValueError(
        "No valid faces found"
    )

face_data = list(
    zip(faces, scores)
)

face_data.sort(
    key=lambda x: x[1],
    reverse=True
)

face_data = face_data[:10]

faces = [
    f for f, s in face_data
]

scores = [
    s for f, s in face_data
]

print(
    f"Selected {len(faces)} sharpest faces"
)

weights = np.array(scores)

weights = weights / np.sum(weights)

fused = np.zeros(
    faces[0].shape,
    dtype=np.float32
)

for face, weight in zip(
    faces,
    weights
):
    fused += (
        face.astype(np.float32)
        * weight
    )

fused = np.clip(
    fused,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    "fused_evidence.jpg",
    fused
)

print(
    f"Evidence fusion complete using {len(faces)} faces"
)
