from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.card_schema import ExtractCardResponse, OCRBox
from app.services.image_preprocess import (
    build_ocr_variants,
    decode_base64_image,
    decode_image_bytes,
)
from app.services.ocr_engine import PaddleOCREngine
from app.services.postprocess import parse_ocr_blocks, select_best_extraction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["ocr"])


async def _extract_image_from_request(
    request: Request,
    image: UploadFile | None,
    image_base64: str | None,
) -> bytes:
    if image is not None:
        payload = await image.read()
        if payload:
            return payload

    if image_base64:
        try:
            decoded = decode_base64_image(image_base64)
            import cv2

            success, encoded = cv2.imencode(".jpg", decoded)
            if not success:
                raise ValueError("Image encode failed")
            return encoded.tobytes()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    if request.headers.get("content-type", "").startswith("application/json"):
        body: dict[str, Any] = await request.json()
        payload = body.get("image_base64")
        if payload:
            try:
                decoded = decode_base64_image(str(payload))
                import cv2

                success, encoded = cv2.imencode(".jpg", decoded)
                if not success:
                    raise ValueError("Image encode failed")
                return encoded.tobytes()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    raise HTTPException(
        status_code=400,
        detail="Provide an image file (multipart) or image_base64 payload.",
    )


@router.post("/extract-card", response_model=ExtractCardResponse)
async def extract_card(
    request: Request,
    image: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
) -> ExtractCardResponse:
    try:
        image_bytes = await _extract_image_from_request(request, image, image_base64)
        decoded_image = decode_image_bytes(image_bytes)
        variants = build_ocr_variants(decoded_image)

        candidate_results = []
        for variant_name, variant_image in variants:
            blocks = PaddleOCREngine.run(variant_image)
            parsed = parse_ocr_blocks(blocks)
            candidate_results.append((variant_name, parsed, blocks))

        selected_variant, parsed, blocks = select_best_extraction(candidate_results)
        logger.info("Selected OCR variant: %s (boxes=%s confidence=%.4f)", selected_variant, len(blocks), parsed.confidence)

        response = ExtractCardResponse(
            card_number=parsed.card_number,
            expiry_date=parsed.expiry_date,
            cardholder_name=parsed.cardholder_name,
            network=parsed.network_type,
            bank=parsed.bank_name,
            confidence=parsed.confidence,
            ocr_boxes=[
                OCRBox(text=b.text, confidence=round(b.confidence, 4), bbox=b.bbox) for b in blocks
            ],
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Card extraction failed")
        raise HTTPException(status_code=500, detail=f"Card extraction failed: {exc}") from exc

