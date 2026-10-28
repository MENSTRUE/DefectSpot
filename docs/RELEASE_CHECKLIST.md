# DefectSpot Day 22 — Release Audit Checklist

**Status:** Documentation preparation checklist, not a declaration of completed production release.

## Before
- [ ] `git status` reviewed, all unrelated changes preserved.
- [ ] Confirm `reports/final_summary.json` and the notebook match the current model assets.
- [ ] Confirm 15 trusted MVTec joblib files, NEU `.pt`, ONNX export and screenshots are present.
- [ ] Verify `.gitignore` is valid; local datasets, `.env` and `node_modules/` ignored.

## Now — reproducibility
- [ ] Backend virtualenv created from `backend/requirements.txt`.
- [ ] Backend `/api/v1/health` and `/api/v1/models` succeed.
- [ ] MVTec `/api/v1/mvtec/categories` reports 15 categories.
- [ ] `backend/tests/test_mvtec_features.py` passes in local environment.
- [ ] Frontend `npm ci`, `npm run build`, `npm run typecheck` pass.
- [ ] NEU known samples infer six ontology classes with proper boxes; no fixed pass/fail.
- [ ] MVTec bottle good/broken_large examples score and render overlay; no fake percentage.
- [ ] Bad upload / missing bank / invalid category cause meaningful errors.
- [ ] Verify screenshot gallery paths open on GitHub.
- [ ] All metrics in docs match committed reports; documented limitations remain prominent.

## Next — release boundaries
- [ ] Review MVTec non-commercial license, NEU source, Ultralytics and weights.
- [ ] Consider per-file hashing and model provenance manifest.
- [ ] Add CI for backend and frontend tests; no fabricated green checks.
- [ ] Decide human review/field validation ownership and data-retention/security policy before deployment.
- [ ] Keep a release tag only after all release-specific checks actually pass.

## Suggested commit

```powershell
cd "D:\porto\100 day\day 22\DefectSpot"
git status
git add README.md AGENTS.md backend/AGENTS.md frontend/AGENTS.md frontend/README.md docs/ .gitignore
git diff --cached --check
git commit -m "docs: finalize DefectSpot dual-pipeline source-of-truth documentation"
git push origin main
```
