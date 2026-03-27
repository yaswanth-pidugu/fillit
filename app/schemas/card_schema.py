from __future__ import annotations

from pydantic import BaseModel, Field


class OCRBox(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[list[float]]


class ExtractCardResponse(BaseModel):
    card_number: str
    expiry_date: str
    cardholder_name: str
    network: str
    bank: str
    confidence: float = Field(ge=0.0, le=1.0)
    ocr_boxes: list[OCRBox] = Field(default_factory=list)


class Base64ExtractRequest(BaseModel):
    image_base64: str

