from __future__ import annotations

import io
import os
from typing import Any, Dict

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image


def _resolve_api_url() -> str:
    default_url = os.getenv("FILLIT_API_URL", "http://127.0.0.1:8000/extract-card")
    try:
        return str(st.secrets.get("api_url", default_url))
    except Exception:
        return default_url


API_URL = _resolve_api_url()

st.set_page_config(page_title="FillIt - Card OCR", layout="wide")
st.title("FillIt - Credit Card OCR")
st.caption("Capture or upload a card image and auto-fill payment fields.")


@st.cache_data(show_spinner=False)
def _draw_boxes(image_bytes: bytes, ocr_boxes: list[dict[str, Any]]) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for item in ocr_boxes:
        box = item.get("bbox", [])
        if len(box) != 4:
            continue
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _call_api(image_bytes: bytes) -> Dict[str, Any]:
    files = {"image": ("card.jpg", image_bytes, "image/jpeg")}
    response = requests.post(API_URL, files=files, timeout=60)
    if response.status_code >= 400:
        detail = response.text
        raise requests.HTTPError(f"{response.status_code} {response.reason}: {detail}", response=response)
    return response.json()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Input")
    camera_capture = st.camera_input("Capture card")
    uploaded_file = st.file_uploader("Or upload card image", type=["jpg", "jpeg", "png"])

    image_bytes = None
    if camera_capture is not None:
        image_bytes = camera_capture.getvalue()
    elif uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

    if image_bytes:
        st.image(image_bytes, caption="Original image", use_container_width=True)

with col2:
    st.subheader("Extracted Details")
    response_payload: Dict[str, Any] | None = None
    if image_bytes and st.button("Run OCR", type="primary"):
        try:
            with st.spinner("Running OCR..."):
                response_payload = _call_api(image_bytes)
            st.success("Extraction complete")
            st.json(response_payload)
        except requests.RequestException as exc:
            st.error(f"API error: {exc}")

    if image_bytes and response_payload and response_payload.get("ocr_boxes"):
        st.image(
            _draw_boxes(image_bytes, response_payload.get("ocr_boxes", [])),
            caption="Detected text boxes",
            use_container_width=True,
        )

    if response_payload:
        st.markdown("### Payment Form")
        st.text_input("Card Number", value=response_payload.get("card_number", ""))
        st.text_input("Expiry Date", value=response_payload.get("expiry_date", ""))
        st.text_input("Cardholder Name", value=response_payload.get("cardholder_name", ""))
        st.text_input("Network", value=response_payload.get("network", ""))
        st.text_input("Bank", value=response_payload.get("bank", ""))

        if st.button("Autofill Simulation"):
            st.success("Payment form auto-filled from OCR response.")

