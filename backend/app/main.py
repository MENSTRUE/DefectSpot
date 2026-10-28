"""DefectSpot FastAPI application.

Two independent inference pipelines:
- NEU-DET: supervised surface-defect detection (YOLOv8n).
- MVTec AD: normal-only handcrafted patch-bank anomaly scoring.

Neither pipeline returns a manufacturing pass/fail decision.
"""

from __future__ import annotations

import base64
import logging
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from app.api.routes.mvtec import router as mvtec_router
from app.inference.anomaly.mvtec import CATEGORIES as MVTEC_CATEGORIES
from app.inference.anomaly.mvtec import MODEL_DIR

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "neu_best.pt"
NEU_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
INFERENCE_LOCK = Lock()

app = FastAPI(
    title="DefectSpot Inference API",
    version="0.2.0",
    description=(
        "Two independent research pipelines: NEU object detection and "
        "MVTec normal-only anomaly scoring. No automatic quality verdict."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# The MVTec router owns /mvtec/categories and /predict/mvtec.
# Never define a second /predict/mvtec route in this file.
app.include_router(mvtec_router)


@lru_cache(maxsize=1)
def get_detector():
    """Load the trusted YOLO checkpoint lazily and verify its class order."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"NEU checkpoint missing: {MODEL_PATH}")

    from ultralytics import YOLO

    detector = YOLO(str(MODEL_PATH))
    names = {int(key): str(value) for key, value in detector.names.items()}
    expected = dict(enumerate(NEU_CLASSES))
    if detector.task != "detect" or names != expected:
        raise RuntimeError(
            f"Checkpoint class contract mismatch: {names!r}; "
            f"task={detector.task!r}"
        )
    return detector


async def read_image(file: UploadFile) -> Image.Image:
    """Read an image in memory with MIME, byte, pixel, and decode checks."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Only JPEG, PNG, and WebP supported")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds 10 MB")

    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(415, "Uploaded bytes are not JPEG, PNG, or WebP")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image too large (pixel limit)")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(422, "Invalid image") from exc


@app.get("/api/v1/health", tags=["System"])
def health():
    banks_present = sum(
        (MODEL_DIR / f"mvtec_patch_{category}.joblib").is_file()
        for category in MVTEC_CATEGORIES
    )
    return {
        "status": "ok",
        "neu_checkpoint_present": MODEL_PATH.is_file(),
        "mvtec_banks_present": banks_present,
        # Files present != every bank has been successfully loaded.
        "mvtec_runtime": "implemented_requires_trusted_bank_validation",
    }


@app.get("/api/v1/models", tags=["System"])
def models():
    return {
        "detection": {
            "task": "object_detection",
            "classes": list(NEU_CLASSES),
            "checkpoint": "models/neu_best.pt",
        },
        "anomaly": {
            "task": "normal_only_patch_anomaly",
            "categories": list(MVTEC_CATEGORIES),
            "status": "runtime_implemented_bank_verified_on_load",
        },
        "warning": "Scores from different research tasks are not directly comparable.",
    }


@app.post("/api/v1/predict/neu", tags=["NEU-DET"])
async def predict_neu(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=1.0),
    include_annotated: bool = Query(False),
):
    """Detect six types of steel defects; retain existing response schema."""
    image = await read_image(file)
    try:
        import numpy as np

        model = get_detector()
        started = time.perf_counter()
        with INFERENCE_LOCK:
            result = model.predict(
                source=np.asarray(image),
                conf=confidence,
                verbose=False,
                imgsz=640,
            )[0]
        elapsed = round((time.perf_counter() - started) * 1000, 2)

        detections = []
        for box in result.boxes if result.boxes is not None else []:
            class_id = int(box.cls.item())
            if not 0 <= class_id < len(NEU_CLASSES):
                raise RuntimeError(f"Unexpected class ID: {class_id}")

            coords = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            detections.append({
                "class_id": class_id,
                "class_name": NEU_CLASSES[class_id],
                "confidence": round(float(box.conf.item()), 6),
                "bbox": dict(zip(("x1", "y1", "x2", "y2"), coords)),
            })

        annotated = None
        if include_annotated:
            canvas = image.copy()
            painter = ImageDraw.Draw(canvas)
            for detection in detections:
                bbox = detection["bbox"]
                painter.rectangle(
                    tuple(bbox[key] for key in ("x1", "y1", "x2", "y2")),
                    outline="#24c4a6",
                    width=3,
                )
                painter.text(
                    (bbox["x1"], max(0, bbox["y1"] - 15)),
                    f"{detection['class_name']} {detection['confidence']:.2f}",
                    fill="#24c4a6",
                )

            buffer = BytesIO()
            canvas.save(buffer, "JPEG", quality=85)
            annotated = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "pipeline": "neu_det",
            "model": "neu_best.pt",
            "width": image.width,
            "height": image.height,
            "detections": detections,
            "detection_count": len(detections),
            "inference_ms": elapsed,
            "annotated_image_base64": annotated,
            "quality_decision": "not_assessed",
            "note": (
                "Detections are visual observations, not validated "
                "manufacturing acceptance decisions."
            ),
        }
    except Exception as exc:
        LOGGER.exception("NEU inference failed")
        raise HTTPException(503, "NEU inference failed; check backend logs") from exc
