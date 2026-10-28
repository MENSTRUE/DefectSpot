# DefectSpot Frontend — React + TypeScript + Vite

**Status: both research inference panels are implemented**, subject to availability of trusted backend artifacts.

## Functionality

- **NEU-DET:** upload steel image, choose request confidence, run `POST /api/v1/predict/neu`, view model-proposed bounding boxes and class/confidence, compare image states and export JSON/annotated JPEG.
- **MVTec AD:** select one of 15 reference categories, upload matching product image, run `POST /api/v1/predict/mvtec`, view raw nearest-neighbor anomaly distance and a coarse 16×16 overlay, export JSON.
- **Overview / Model Insights:** static research results are drawn from repo reports; do not confuse them with newly computed browser statistics.
- **Quality verdict:** deliberately `Not assessed` in both modes; operators must review outputs.

## Setup

Use Node.js/npm compatible with the committed Vite version. In one terminal start FastAPI in `backend/` on port 8000. In another:

```powershell
cd "D:\porto\100 day\day 22\DefectSpot\frontend"
npm ci
npm run build
npm run typecheck
npm run dev
```

Open `http://127.0.0.1:5173`. `vite.config.ts` proxies `/api` to `http://127.0.0.1:8000` in development. API calls default to `/api/v1`; override only when correctly configured through `VITE_API_BASE_URL`.

## Model/output interpretation

NEU confidence is an object detection model score for one of six steel-defect types. MVTec image anomaly score is **uncalibrated distance**, not probability; overlay heatmap uses **per-image min/max display normalization**, so images' colors must not be compared on an absolute scale. MVTec heatmap is spatially coarse (16×16). Neither task offers acceptance/rejection rules.

## Test checklist

1. Backend health returns status and model presence; presence alone does not validate a checkpoint.
2. Known NEU sample returns model boxes and annotation; 0 boxes remains an allowed output.
3. Known MVTec bottle good and defect samples return raw score, grid and optional overlay. Reference demonstration: 0.7583 vs 0.9847 for one test pair, **not** an accuracy metric.
4. Export JSON excludes unnecessary embedded image when implemented; inspect saved file validity.
5. `npm run build` and `npm run typecheck` succeed; confirm keyboard/tab navigation and error states.

## Troubleshooting

- API Offline: start backend on port 8000, verify `/api/v1/health`, check dev proxy.
- HTTP 503: verify trusted model presence, compatible dependencies and bank schema, inspect backend logs.
- HTTP 415/413/422: invalid image type, size, pixel count or category.
- TypeScript `ImportMeta.env` error: ensure `src/vite-env.d.ts` exists.
- Unused variable error: remove unused hook/state/import; don't weaken strict TypeScript checks.
- Inconsistent heatmap scaling: normal—display normalization is per image. Compare **raw scores within one category**, with proper validation.

## Source and licensing

See root [`README.md`](../README.md), [`AGENTS.md`](AGENTS.md), data/evaluation evidence under `reports/`, and third-party license caveats. This is a research prototype, not a production quality-control release.
