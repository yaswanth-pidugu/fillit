from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import List

import numpy as np

from app.utils.constants import OCR_LANG, OCR_USE_ANGLE_CLS, OCR_USE_GPU


@dataclass
class OCRTextBlock:
    text: str
    confidence: float
    bbox: list[list[float]]


class PaddleOCREngine:
    """Thread-safe lazy singleton wrapper for PaddleOCR."""

    _model = None
    _lock = Lock()

    @classmethod
    def get_model(cls):
        if cls._model is not None:
            return cls._model

        with cls._lock:
            if cls._model is not None:
                return cls._model
            try:
                from paddleocr import PaddleOCR
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "PaddleOCR import failed. Install requirements and verify environment setup."
                ) from exc

            cls._model = PaddleOCR(
                use_angle_cls=OCR_USE_ANGLE_CLS,
                lang=OCR_LANG,
                use_gpu=OCR_USE_GPU,
                show_log=False,
            )
        return cls._model

    @classmethod
    def run(cls, image: np.ndarray) -> List[OCRTextBlock]:
        model = cls.get_model()
        raw = model.ocr(image, cls=True)

        blocks: List[OCRTextBlock] = []
        if not raw or len(raw) == 0 or raw[0] is None:
            return blocks
        lines = raw[0]
        if not isinstance(lines, list):
            return blocks

        for item in lines:
            if not item or len(item) < 2:
                continue
            bbox = item[0]
            text_info = item[1]
            if not text_info or len(text_info) < 2:
                continue
            text = str(text_info[0]).strip()
            confidence = float(text_info[1])
            if text:
                blocks.append(OCRTextBlock(text=text, confidence=confidence, bbox=bbox))

        return blocks

