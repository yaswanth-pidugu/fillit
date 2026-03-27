from __future__ import annotations

import base64
from typing import List, Tuple

import cv2
import numpy as np


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes.")
    return image


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode plain base64 or data URI image payloads."""
    value = image_base64.strip()
    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]
    image_bytes = base64.b64decode(value)
    return decode_image_bytes(image_bytes)


def resize_keep_aspect(image: np.ndarray, target_width: int = 1024) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 2 or max_height < 2:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def perspective_correction(image: np.ndarray) -> np.ndarray:
    """Try to detect a card-like quadrilateral and rectify it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            try:
                transformed = _four_point_transform(image, pts)
            except cv2.error:
                continue
            if transformed.shape[0] > 50 and transformed.shape[1] > 50:
                return transformed
    return image


def preprocess_for_ocr(image: np.ndarray, apply_perspective: bool = True) -> np.ndarray:
    base_image = resize_keep_aspect(image)
    if apply_perspective:
        base_image = perspective_correction(base_image)

    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    return thresholded


def build_ocr_variants(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Generate multiple OCR-ready variants to improve robustness across capture conditions."""
    resized = resize_keep_aspect(image)
    corrected = perspective_correction(resized)

    variants: List[Tuple[str, np.ndarray]] = [
        ("raw", resized),
        ("raw_perspective", corrected),
        ("threshold", preprocess_for_ocr(resized, apply_perspective=False)),
        ("threshold_perspective", preprocess_for_ocr(resized, apply_perspective=True)),
    ]
    return variants


def draw_ocr_boxes(image: np.ndarray, boxes: list[list[list[float]]]) -> np.ndarray:
    output = image.copy()
    for box in boxes:
        if len(box) != 4:
            continue
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return output


