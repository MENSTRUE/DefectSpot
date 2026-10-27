# DefectSpot — Day 22 Frontend

Research-grade React + TypeScript + Vite dashboard for two **scientifically separate** computer-vision tracks.

## Current implementation (source of truth)

- **NEU-DET**: real inference using `POST /api/v1/predict/neu` on the Day 22 FastAPI backend. Upload JPG/PNG/WebP (<=10MB), set confidence, render annotated backend JPEG, inspect bounding boxes/classes/confidences, export JSON (excluding large Base64) and JPEG.
- **MVTec AD**: displays categories, research context, and a **clearly disabled runtime**. It does not send fabricated predictions: the current backend intentionally returns HTTP 501 pending verification of `.joblib` bank schema and the exact training feature extractor.
- Research metrics are **static values reported in the existing repository reports**, not measurements made by the running browser.
- A detection is not a manufacturing pass/fail decision; `quality_decision` remains `not_assessed`.

## Setup on Windows

From `D:\porto\100 day\day 22\DefectSpot`:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd "D:\porto\100 day\day 22\DefectSpot\frontend"
npm install
npm run build
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite dev server proxies `/api` to `http://127.0.0.1:8000`. The frontend does not require changing the backend CORS. If port 8000 is occupied by PPE Guard, stop PPE Guard first or reconfigure Vite's proxy to the port serving DefectSpot.

## End-to-end test

1. Verify `http://127.0.0.1:8000/api/v1/health` responds and `neu_checkpoint_present` is true.
2. In **AI Inspection**, select **NEU-DET**, upload a NEU steel-defect photo, and run at `confidence=0.25`.
3. Verify annotated image, actual `detections[]` list, and export files. Zero detections at one threshold is possible and should not be replaced with fake data.
4. Switch to **MVTec AD**. The UI should say runtime validation is pending and **must not** show made-up anomaly heatmaps or scores.
5. Run `npm run build` and inspect for TypeScript/build errors.

## Research evidence and limits

`reports/final_summary.json` reports MVTec macro image AUROC 0.81172 and macro **coarse** pixel AUROC 0.88824 (16x16 grid; not official MVTec benchmark protocol); NEU held-out test 270 images, precision 0.7189, recall 0.7365, mAP@50 0.7852, mAP@50–95 0.4603. These two metrics families are not interchangeable. Training dataset sources are documented in the root README; do not upload raw datasets into this repo. NEU near-duplicate leakage may remain and MVTec is non-commercial licensed. No production-quality assurance claim is made.

## Troubleshooting

- **API Offline / Failed to fetch**: check FastAPI, port 8000, `/api/v1/health`, and Vite proxy.
- **HTTP 503**: verify `models/neu_best.pt` is present and class mapping matches the backend contract.
- **HTTP 415 / 413 / 422**: choose a supported, valid image <=10 MB and within server pixel limits.
- **Blank/zero detections**: verify sample domain and threshold, inspect backend logs, and compare a known test sample. Never assume no boxes means no defect.
- **MVTec disabled**: intentional until exact feature pipeline and category banks are confirmed. Never substitute an approximate preprocessing algorithm.
- **No `node_modules`**: run `npm install` in `frontend/`, then `npm run build`.

## License and distribution

Repository `LICENSE` covers original code where applicable; pretrained weights, MVTec AD and NEU-DET may carry distinct rights. Check those sources before model redistribution or commercial deployment. Image uploads are processed by the local backend; avoid handling sensitive factory images in untrusted deployments.
