# Frontend AGENTS — DefectSpot

Scope: `frontend/`. Follow root `AGENTS.md` first.

## UI and API contracts
- React 18, TypeScript, Vite; current entrypoints are `src/App.tsx` and `src/api.ts`, with MVTec component at `src/components/inspection/MvtecInspection.tsx`.
- Maintain **separate experiences** for NEU object detection and MVTec normal-only anomaly; do not reuse confidence threshold, box table or class counts as a meaning for MVTec distances.
- Use API responses, not local simulated inference. Vite dev `/api` proxy forwards to backend 8000; keep `src/vite-env.d.ts` for `import.meta.env` typing.
- Show `not_assessed` rather than a quality verdict. Warn that anomaly distances are raw/un-calibrated and heatmaps are 16×16 and display-normalized per image.
- Preserve 10MiB JPEG/PNG/WebP client checks and backend error display; client checks are not a security boundary.

## Content and design
- Professional dark industrial dashboard with clear contrast; keyboard-operable file controls; responsive layout; consistent spacing and language.
- Separate research metrics displayed from `reports/` from live browser inference timings; never show any metric as calculated live unless actually fetched/calculated.
- Keep screenshot URLs and UI documentation current; do not link filenames that are not in repo.
- Do not leave stale text stating MVTec `501` or disabled when its backend is implemented.
- Avoid inert buttons, ignored state variables and unused imports: TS build uses strict unused rules.

## Validation

```powershell
cd frontend
npm ci
npm run build
npm run typecheck
npm run dev
```

Test NEU upload/display/export, MVTec category/file/overlay/JSON export, missing API, unsupported files, 0 detections and responsive layout. Use `npm ci` with committed `package-lock.json`.

## Boundaries
No client-side pass/fail heuristics, fake probabilities, local secrets, hardcoded user images, arbitrary model deserialization or claims of manufacturing certification. Never hide a model failure behind a success state.
