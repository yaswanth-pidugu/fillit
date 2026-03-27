import numpy as np

from app.services.image_preprocess import build_ocr_variants


def test_build_ocr_variants_returns_expected_passes() -> None:
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    variants = build_ocr_variants(image)
    names = [name for name, _ in variants]

    assert "raw" in names
    assert "raw_perspective" in names
    assert "threshold" in names
    assert "threshold_perspective" in names
    assert len(variants) == 4

