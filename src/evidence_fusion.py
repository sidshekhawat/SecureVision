import numpy as np
import cv2
from pathlib import Path
from retinaface import RetinaFace

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
INPUT_DIR = ROOT / "extracted_faces"

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input directory not found: {INPUT_DIR}"
    )

faces = []
scores = []

images = (
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg")) +
    list(INPUT_DIR.glob("*.png"))
)

print(
    f"Found {len(images)} faces"
)

if len(images) == 0:
        raise ValueError(
            "No face images found."
        )

for img_path in images:

    img = cv2.imread(str(img_path))

    if img is None:
        continue

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

save_path = OUTPUT_DIR / "fused_evidence.jpg"

cv2.imwrite(
    str(save_path),
    fused
)

print(
    f"Evidence fusion complete using {len(faces)} faces."
)

print(
    f"Saved fused image to {save_path}"
)
