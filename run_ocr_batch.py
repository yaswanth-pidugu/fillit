from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.services.image_preprocess import preprocess_for_ocr
from app.services.ocr_engine import PaddleOCREngine
from app.services.postprocess import parse_ocr_blocks


def run_batch(images_dir: Path, repeats: int) -> None:
    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.jpeg"), *images_dir.glob("*.png")])
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} images in {images_dir}")
    for run_idx in range(1, repeats + 1):
        print(f"\n=== Run {run_idx}/{repeats} ===")
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"{image_path.name}: read_failed")
                continue

            processed = preprocess_for_ocr(image, apply_perspective=True)
            blocks = PaddleOCREngine.run(processed)
            result = parse_ocr_blocks(blocks)

            payload = {
                "file": image_path.name,
                "card_number": result.card_number,
                "expiry_date": result.expiry_date,
                "cardholder_name": result.cardholder_name,
                "network": result.network_type,
                "bank": result.bank_name,
                "confidence": result.confidence,
                "ocr_blocks": len(blocks),
            }
            print(json.dumps(payload, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FillIt OCR pipeline on all images in a folder.")
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Path to image directory")
    parser.add_argument("--repeats", type=int, default=3, help="How many repeated runs to execute")
    args = parser.parse_args()

    run_batch(args.images_dir, max(1, args.repeats))


if __name__ == "__main__":
    main()