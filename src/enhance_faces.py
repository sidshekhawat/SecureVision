"""
enhance_faces.py

Purpose:
Enhances extracted face images using GFPGAN.

Workflow:
1. Reads face crops
2. Restores facial details
3. Improves sharpness and clarity
4. Saves results into enhanced_faces/

Technology:
GFPGAN
"""
import os
import cv2
from gfpgan import GFPGANer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = ROOT / "extracted_faces"
OUTPUT_FOLDER = ROOT / "enhanced_faces"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        f"Input folder not found: {INPUT_FOLDER}"
    )

if not any(INPUT_FOLDER.iterdir()):
    print("No images found to enhance.")
    exit()

restorer = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    upscale=2,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

for file in os.listdir(INPUT_FOLDER):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = INPUT_FOLDER / file

        img = cv2.imread(str(img_path))

        _, _, restored = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        save_path = OUTPUT_FOLDER / file

        cv2.imwrite(
            str(save_path),
            restored
        )

        print(f"Enhanced: {file}")

print("Done!")