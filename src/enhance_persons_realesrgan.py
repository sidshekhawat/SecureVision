"""
enhance_persons_realesrgan.py

Purpose:
Enhances extracted person images using Real-ESRGAN.

Workflow:
1. Reads person crops
2. Upscales image resolution
3. Improves image quality
4. Saves results into enhanced_persons/

Technology:
Real-ESRGAN
"""
import cv2
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Model
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

weights_path = Path("models/RealESRGAN_x4plus.pth")

if not weights_path.exists():
    raise FileNotFoundError(
        f"Model weights not found: {weights_path}"
    )

upsampler = RealESRGANer(
    scale=4,
    model_path=str(weights_path),
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

input_dir = Path("extracted_persons")
if not input_dir.exists():
    raise FileNotFoundError(
        f"Input directory not found: {input_dir}"
    )
output_dir = Path("enhanced_persons")
output_dir.mkdir(exist_ok=True)

images = (
    list(input_dir.glob("*.jpg")) +
    list(input_dir.glob("*.jpeg")) +
    list(input_dir.glob("*.png"))
)

print(f"Found {len(images)} images to enhance")
if len(images) == 0:
    print("No images found in extracted_persons/")
for img_path in images:
    img = cv2.imread(str(img_path))

    if img is None:
        continue

    try:

        output, _ = upsampler.enhance(
            img,
            outscale=2
        )

        save_path = output_dir / img_path.name

        cv2.imwrite(
            str(save_path),
            output
        )

        print(f"Enhanced: {img_path.name}")

    except Exception as e:

        print(
            f"Failed to process {img_path.name}: {e}"
        )
    print(
        f"\nFinished processing {len(images)} images"
    )