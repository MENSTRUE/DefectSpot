# DefectSpot — English Technical Companion

**Project:** Day 22/100, Industrial Visual Quality Intelligence.  
**Primary detailed documentation:** [`README.md`](../../README.md). **Agent/contributor contract:** [`AGENTS.md`](../../AGENTS.md).

## Research questions and implementation

Two **scientifically distinct** tasks share one FastAPI + React demo:

- **NEU-DET**, supervised YOLOv8n on six steel defect categories, provides class/confidence/bounding boxes, optional annotated image. Live model is `models/neu_best.pt`; `neu_best.onnx` is an optional export.
- **MVTec AD**, 15 category-specific normal-only banks with handcrafted 14D patch features, provides raw nearest-neighbor anomaly distance and coarse 16×16 heatmap, not a defect probability or defect-type classification.

Both responses intentionally say **not assessed** for manufacturing acceptance.

## Before → Now → Next

| Before | Now | Next |
|---|---|---|
| Source audits, VOC annotations, grouped train/val/test split, normal-only reference learning | Two live local inference endpoints, React dashboard, screenshots, versioned metrics/reports | Continuous tests, calibrated category thresholds, field evaluation, pinned reproducibility, secure deployment |

## Evidence and metrics

Read `reports/final_summary.json` as the project summary, `reports/neu_heldout_metrics.json` for 270-image NEU test and `reports/mvtec_heldout_metrics.csv` for per-category anomaly evaluation.

| Pipeline | Metric | Recorded held-out value |
|---|---|---:|
| NEU-DET | Precision | 0.718906 |
| NEU-DET | Recall | 0.736539 |
| NEU-DET | mAP50 | 0.785180 |
| NEU-DET | mAP50–95 | 0.460330 |
| MVTec | Macro image AUROC | 0.811720 |
| MVTec | Macro **16×16 coarse pixel** AUROC | 0.888239 |

These are research metrics, not live browser or product-release guarantees. A single MVTec bottle good/broken_large pair scored 0.7583/0.9847, which is only a functional sanity check.

## Dataset lineage and integrity

NEU recorded mount: `/kaggle/input/datasets/wafabila/neu-det`, grouped split 1259/270/270, one exact image duplicate excluded, no missing class per split; near-duplicate risk persists. MVTec recorded mount `/kaggle/input/datasets/wafabila/mvtec-ad-dataset`, source can be FiftyOne `samples.json` with `data/data_*`, 15 categories/1725 test images. Bank trained only on normal images, no exact test overlap in category hash audit, 0 missing masks in recorded integrity report. See `reports/neu_split_audit.json`, `reports/mvtec_integrity.json` and notebook.

## Reproducibility and operations

Research configuration: `reports/environment.json` (seed 42, Python 3.13.15, Torch 2.11+cu128, sklearn 1.6.1, OpenCV 4.14.0, NEU up to 70 epochs/patience 20, MVTec 128px/16×16). Dataset access is needed to retrain; the local app uses committed artifacts.

```powershell
# Terminal A
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Terminal B, from repository root
cd frontend
npm ci
npm run build
npm run dev
```

Browse `http://127.0.0.1:8000/docs` and `http://127.0.0.1:5173`. Endpoints: `GET /api/v1/health`, `/models`, `/mvtec/categories`; `POST /predict/neu` and `/predict/mvtec` (all under `/api/v1`). Test with `pytest backend/tests/test_mvtec_features.py -q` in correctly configured environment and `npm run build/typecheck`.

## Limitations and release decisions

- No calibrated pass/fail; NEU confidence and MVTec distances are incomparable.
- MVTec colors are min/max scaled per image, heatmap is 16×16 not pixel segmentation.
- Sensitive to domain shift; six-class NEU performance varies greatly by class.
- Local demo is not production-hardened; add auth, TLS, throttling, monitoring and model provenance for deployment.
- Licensing: MIT covers original code; MVTec CC BY-NC-SA non-commercial; NEU source and Ultralytics/pretrained weights require separate rights checks.

## Troubleshooting / FAQ

**API offline?** Start backend 8000, check `/health`, Vite proxy. **HTTP 503?** Check checkpoint/bank presence and contract, logs. **MVTec 501?** Old backend still running. **Build `import.meta.env` error?** Ensure `src/vite-env.d.ts`. **Is raw score a probability?** No. **Is this PatchCore?** No. **Is ONNX deployed?** No, current NEU runtime uses `.pt`. **Can I market this as automatic factory QA?** Not without external validation, thresholds, governance and license review.

For full repository structure, roadmap, test matrix, source claims, deployment checklist and detailed FAQ consult the **root README**, which is the canonical project document.
