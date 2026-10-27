"""Add this router to the existing DefectSpot FastAPI app. No NEU routes changed."""
import io
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from app.inference.anomaly.mvtec import CATEGORIES, MODEL_DIR, predict_mvtec

router = APIRouter(prefix="/api/v1", tags=["MVTec anomaly"])
logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@router.get("/mvtec/categories")
def mvtec_categories():
    return {"categories": [
        {"name": c, "checkpoint_available": (MODEL_DIR / f"mvtec_patch_{c}.joblib").is_file()}
        for c in CATEGORIES
    ], "inference_enabled": True,
        "note": "Anomaly scores are uncalibrated distances and not quality pass/fail outcomes."}


@router.post("/predict/mvtec")
async def mvtec_predict(
    file: UploadFile = File(...),
    category: str = Query(..., description="One of 15 MVTec categories"),
    include_overlay: bool = Query(True),
):
    if category not in CATEGORIES:
        raise HTTPException(422, detail=f"Unknown MVTec category: {category}")
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, detail="Upload a JPEG, PNG or WebP image")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, detail="Image exceeds 10 MB")
    if not raw:
        raise HTTPException(400, detail="Empty upload")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported image bytes")
            if im.width * im.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, detail="Image pixel count exceeds limit")
            im.load()
            image = ImageOps.exif_transpose(im).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, detail="Unreadable image") from exc
    try:
        output = predict_mvtec(image, category, with_overlay=include_overlay)
    except FileNotFoundError as exc:
        raise HTTPException(503, detail="Category model not installed") from exc
    except (ValueError, RuntimeError) as exc:
        logger.exception("MVTec contract check or scoring failed")
        raise HTTPException(503, detail="MVTec bank incompatible; inspect backend logs") from exc
    output["image"]["filename"] = Path((file.filename or "upload.jpg").replace("\\", "/")).name[:140]
    return output
