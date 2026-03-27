# FillIt - Credit Card OCR

Production-oriented OCR pipeline for extracting card fields from camera/uploaded images using PaddleOCR + FastAPI + Streamlit.

## Features
- OpenCV preprocessing: resize, grayscale, CLAHE, adaptive threshold, perspective correction
- PaddleOCR (`lang='en'`, `use_angle_cls=True`, `use_gpu=False`) with singleton model reuse
- Postprocessing with regex + heuristics for PAN, expiry, and name
- Strict validation: Luhn + network length checks + confidence scoring
- BIN-based bank detection + network detection using strict prefix/range rules
- FastAPI endpoint `/extract-card` supports multipart and base64
- Streamlit demo with capture/upload + detected box overlay + autofill simulation

## Run
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

```powershell
streamlit run streamlit_app/app.py
```

## Test
```powershell
python -m pytest tests -q
```

# Credit Card OCR Setup Guide

## 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 2. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 3. Install dependencies (STRICT ORDER)

```bash
pip install numpy==1.26.4
pip install opencv-python==4.6.0.66
pip install paddlepaddle==2.6.2
pip install paddleocr==2.7.3
pip install shapely pyclipper Pillow
```

---

## 4. IMPORTANT RULES

- Do NOT install `opencv-python-headless`
- Do NOT install `opencv-contrib-python`
- Do NOT upgrade numpy (must stay < 2.0)

---

## 5. Verify Setup

```bash
python
```

```python
import numpy
print(numpy.__version__)
```

Expected:
```
1.26.4
```

---

