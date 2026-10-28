# Backend AGENTS — DefectSpot

Scope: `backend/`. Follow root `AGENTS.md` first.

## Runtime routing and artifacts
- `app/main.py` is the FastAPI entrypoint: `/api/v1/health`, `/api/v1/models`, `/api/v1/predict/neu`.
- `app/api/routes/mvtec.py` owns `/api/v1/mvtec/categories` and `/api/v1/predict/mvtec`. Do not duplicate these routes.
- Resolve assets from project root, never relative to current working directory. `models/neu_best.pt` must be trusted and match six-class order. `models/mvtec_patch_<category>.joblib` are trusted research outputs; never accept user-supplied serialized models.
- Preserve image file checks (JPEG/PNG/WebP, max 10 MiB, max 16MP, EXIF orientation), meaningful client errors and diagnostics. Internal exception details belong in logs, not response bodies.
- Neither inference path may return automatic manufacturing accept/reject. NEU `quality_decision` and MVTec `decision` remain `not_assessed`.

## MVTec scientific contract
- `app/inference/anomaly/mvtec.py` must match notebook `patch_features/score_image` exactly: RGB -> resize 128 OpenCV INTER_AREA -> 16×16 patches -> Lab/grayscale/Sobel/Laplacian/quantile 14D -> per-bank normalization -> nearest neighbor -> top 13 mean -> grid map.
- No guessed thresholds, fake probabilities, cross-category score normalization or high-resolution segmentation claims. Display heatmap is scaled **per image**.
- Verify category, schema, number of features and model presence; catch load failures safely. Validate against notebook and trusted samples after dependency changes.

## NEU scientific contract
- Strict class order from root AGENTS, supported boxes `xyxy`, confidence 0–1, pixel width/height, annotated base64 optional.
- Primary live backend uses Ultralytics `.pt`; `.onnx` is not automatically active. Do not change ontology by label alias.

## Tests, dependencies and API

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests/test_mvtec_features.py -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Use `/docs` for contract inspection and real image smoke tests. `requirements.txt` currently uses ranges; pin a separate reproducibility lock before production. Adding tests is preferred over weakening runtime checks. New API fields require corresponding React type updates and docs.

## Deployment policy
Local research only. Harden auth, CORS, timeouts, rate limits, TLS, model trust, logs, uploads and legal rights before external hosting. No raw uploaded data retention by default. Do not call the demo production-ready.
