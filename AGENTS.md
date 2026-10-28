# AGENTS.md — DefectSpot Repository Engineering Contract

> Applies to the **entire repository**. More specific `backend/AGENTS.md` or `frontend/AGENTS.md` rules apply inside their directories. Humans, AI coding assistants, automated agents and contributors must follow this contract.

## 1. Mission and scope

DefectSpot (Day 22/100) is a research-grade **two-pipeline industrial visual inspection prototype**. Maintain science/data provenance, transparent UI, reproducible artifacts, security and honest claim boundaries. **Never pretend this is certified manufacturing QC.**

The two independent tasks must remain logically separate:

| Pipeline | Source of truth | Models | Outputs |
|---|---|---|---|
| NEU supervised defect detection | `notebooks/defectspot.ipynb`, `reports/neu_heldout_metrics.json`, `reports/neu_split_audit.json` | `models/neu_best.pt`, optional `models/neu_best.onnx` | Six classes, XYXY boxes, confidence, optional annotated image |
| MVTec normal-only anomaly | Same notebook, `reports/mvtec_heldout_metrics.csv`, `reports/mvtec_summary.json`, `reports/mvtec_integrity.json` | 15 `models/mvtec_patch_<category>.joblib` | Uncalibrated raw distance, 16×16 map, display overlay |

## 2. Read first; facts outrank prose

Before editing:

1. Inspect `git status`, tree and changed files; **never overwrite user changes** without comparing diffs.
2. Read root `README.md`, appropriate scoped `AGENTS.md`, and actual files to be modified.
3. Read `reports/final_summary.json`, `reports/environment.json`, and task-specific reports before editing metrics.
4. Inspect the executed notebook for training features, split definition, class mapping and checkpoint selection.
5. Check screenshot filenames in `assets/screenshots/` rather than guessing paths.
6. List changes and verify they preserve already working NEU and MVTec behavior.

**Evidence order:** real executable source/notebook and generated artifacts > versioned reports > observed repo file structure > docs/screenshot descriptions > conjecture. When evidence conflicts, document it; do not silently “correct” metrics.

## 3. Before → Now → Next discipline

All substantial contributions must record:

- **Before:** observed behavior, source revision, problem, tests and baseline evidence.
- **Now:** exact files changed, user-visible effect, actual runtime/test results; distinguish verified from unverified.
- **Next:** unresolved risks, next tests, staged improvements, no promises of unsupported completion.

Do not label a feature “complete” solely because its button renders or a single example succeeded. Likewise, do not claim a pipeline is disabled when the committed runtime implements it.

## 4. Non-negotiable scientific integrity

- **Never fabricate** accuracy, AP, AUROC, recall, FPS, model weights, images, annotations, confusion matrices or benchmark results.
- **Never retrain** to satisfy a documentation/UI request. If retraining is requested, retain existing artifacts and make the new experiment versioned and auditable.
- Never tune parameters on held-out test data then publish that same test as independent evidence.
- NEU splits are grouped by exact image hash and filename family. Preserve the source test boundary and manifests; watch for near-duplicate leakage.
- MVTec reference banks are fit **only** on training good samples; evaluation uses test good/defective and masks. Never train on test defects or masks.
- Do not combine NEU steel class logits/confidences with MVTec nearest-neighbor distances or fabricate a common quality score.
- No pass/fail, “defective with 98% confidence”, industrial certification, or legal compliance statement from these uncalibrated outputs.
- MVTec coarse pixel AUROC is from a **16×16 patch grid** and not pixel-precise localization or official benchmark protocol.
- Support figure claims with exact `reports/` path and protocol in docs and release notes.

## 5. Model / preprocessing contracts

### NEU-DET

- Exact class order: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- Live Python backend loads **trusted** `models/neu_best.pt` via Ultralytics; `neu_best.onnx` is an optional export, not the live backend path.
- Keep XYXY absolute-pixel coordinates, structured `detections[]`, image dimensions, `inference_ms`, `annotated_image_base64`, `quality_decision: not_assessed`.
- Default confidence 0.25 is a UI/request choice, not research mAP threshold.

### MVTec AD

- Exact categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.
- **Reference implementation:** notebook `patch_features()` and `score_image()` in `notebooks/defectspot.ipynb`.
- Resize RGB to 128×128 using `cv2.INTER_AREA`; grid 16×16, 256 8×8 patches, feature size **14**: Lab 6, gray 2, gradient magnitude 2, |Laplacian| 1, gray percentiles 3.
- Apply bank-specific mu/sigma; score each patch with `nn.kneighbors()`. Raw image score = average **top ceil(5% of 256) = 13** distances.
- Verify `nn`, `mu`, `sigma`, `resize`, `grid`, `category` contract and trusted origin **before** load/use.
- Heatmap is per-image display min/max scaling and coarse diagnostic; never use its color as comparable absolute anomaly severity across images.
- Do not replace handcrafted features with deep embeddings or call the method “PatchCore” without a new experiment.

## 6. Code and architecture

- Preserve small, readable modules; typed interfaces and docstrings for public contracts. Avoid massive single-function patches when modularization is safe.
- Scope changes: `backend/` API/inference/tests, `frontend/` UI/types/client, `reports/` immutable executed evidence, `docs/` docs, `assets/` visuals.
- Avoid duplicate FastAPI endpoints, especially `/api/v1/predict/mvtec` (owned by `backend/app/api/routes/mvtec.py`).
- Preserve the Vite `/api` proxy and locally used `127.0.0.1:5173`, `127.0.0.1:8000` unless a change is documented and tested.
- Never hardcode Windows paths into generic app logic; root model resolution is based on `Path(__file__)`.
- Do not add placeholders, dummy predictions, hard-coded “live” scores, fake test files or unused controls to pass a demo.
- For new optional features, clearly show unavailable state until fully implemented/tested.
- Avoid introducing new dataset folders into GitHub; notebook input source remains on Kaggle, with manifests in `reports/`.
- Do not claim features exist because an empty directory (`.gitkeep`) exists.

## 7. Security and data boundaries

- **Never deserialize untrusted `.joblib`, pickle or `.pt` uploads**; artifacts can execute code at load time. Treat pretrained model sources as external dependencies with separate terms.
- Enforce image MIME, actual decode, 10 MiB size limit, 16MP pixel cap. Preserve EXIF-aware RGB conversion.
- No secrets, AWS/API keys, credentials, raw private factory images, `.env` or local dataset dumps in commits.
- Avoid logging raw image bytes, base64 images or sensitive paths from users.
- Do not expose local development endpoints publicly without authentication, TLS/reverse proxy controls, rate limiting and governance.
- Dependency upgrades must be justified; test Ultralytics, Torch, OpenCV, sklearn compatibility before changing constraints.

## 8. Testing matrix (minimum before merge)

| Change type | Required evidence |
|---|---|
| Docs only | Markdown links exist; screenshot paths resolve; claims trace to current reports; no stale 501 claim. |
| MVTec feature changes | `python -m pytest backend/tests/test_mvtec_features.py -q` in appropriately configured environment; compare feature output against notebook; run one trusted bottle sample. |
| NEU detection changes | Health, model contract, known sample, detection schema, annotated image, error handling; check original six-class results and no unexpected class-order changes. |
| Frontend changes | `npm ci`, `npm run build`, `npm run typecheck`, navigation/keyboard controls, both upload flows, empty/error states and exports. |
| API / cross-cutting changes | Both endpoint smoke tests; malformed file, missing model, invalid MVTec category, unknown MIME and oversized input. |
| Training / evaluation changes | Re-run dataset audit, group splits, seed tracking and held-out metrics; preserve older run for comparison. |

**Do not claim a test ran if it was only read or reasoned about.** Record command, environment and result. Optional features failing do not license fake success.

## 9. Reproducibility requirements

- Preserve `reports/environment.json` (research Python 3.13.15, PyTorch 2.11+cu128, sklearn 1.6.1, OpenCV 4.14.0, seed 42).
- Differentiate research environment from local frontend/backend environment; version ranges are not exact lockfiles.
- Use dataset version, source paths, file hash, preprocessing, class order, training setup, split manifests, metrics and checkpoint provenance.
- Use validation for selection, held-out test for final reporting. No silent class-remapping/augmentation-family splitting changes.
- Add hashes/manifests for future releases; never overwrite `reports/final_summary.json` without new executed evidence.

## 10. Documentation and external communications

- Maintain root README in coherent professional English, and Bahasa Indonesia technical companion in `docs/id/`.
- Screenshots in `assets/screenshots/` are documentation of UI state; verify file names and do not call them ground truth.
- Any README metric must say source, split/protocol and whether aggregate/mean/single sample.
- Keep `frontend/README.md` synchronized with current **implemented MVTec inference**, not the retired HTTP 501 placeholder.
- When asked for “full code,” supply **full files** and exact replacement paths, not incomplete add-this-line fragments.
- When creating release notes or a social post, distinguish problem, implemented solution, evidence, limits and next action.
- Track MVTec dataset non-commercial CC BY-NC-SA restrictions and unresolved NEU source licensing; the project MIT license never overwrites third-party rights.

## 11. Commit and release conventions

Use small, reviewable conventional commit messages:

```text
docs: document DefectSpot dual-pipeline research and operations
fix: align MVTec UI state with current runtime
feat: add regression tests for MVTec score parity
chore: pin serving environment and workflow checks
```

Before pushing: `git status`, `git diff --check`, relevant builds/tests, scan for secrets/large raw datasets, confirm model artifacts are unchanged unless intentionally versioned. When PRs are used, include Before/Now/Next, evidence and risk register. Never force push merely to hide mismatched files.

## 12. Do not touch without explicit need

- `notebooks/defectspot.ipynb`, trained `models/*.pt`, `models/*.onnx`, `models/*.joblib`, existing `reports/` scientific outputs and source screenshot files.
- The LICENSE body without a rights review.
- User-provided UI code that is already functional, unless refactoring is requested and regression-tested.

## 13. Definition of done

A change is done only when (1) it solves the specified task, (2) it is supported by source and tested where feasible, (3) it preserves existing working features, (4) documentation reflects the exact observed state, (5) no metrics/claims are fabricated, and (6) unresolved limitations are stated explicitly.
