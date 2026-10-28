# DefectSpot — Panduan Teknis Bahasa Indonesia

**Day 22/100 | Dua pipeline inspeksi visual | Prototipe penelitian**

> Dokumentasi utama yang paling detail: [`README.md`](../../README.md). Aturan kontribusi: [`AGENTS.md`](../../AGENTS.md). Panduan ini menjelaskan praktik penggunaan dan batasan dalam Bahasa Indonesia tanpa mengubah hasil penelitian.

## Ringkasan masalah

DefectSpot menjawab dua pertanyaan **berbeda** dalam quality inspection:

1. **NEU-DET:** bagian mana dari permukaan baja yang terdeteksi memiliki salah satu dari enam jenis cacat? Model YOLOv8n mengeluarkan *bounding box*, kelas, dan confidence.
2. **MVTec AD:** seberapa jauh pola visual gambar suatu kategori produk dari referensi normal? Sistem menghasilkan **raw anomaly distance** dan heatmap **kasar 16×16** dari bank nearest-neighbor. Tidak memprediksi jenis cacat sebagai kelas.

Keduanya tidak boleh disatukan menjadi satu angka “quality score” tanpa metodologi validasi terpisah. **Tidak tersedia keputusan otomatis pass/fail.**

## Before → Now → Next

| Tahap | Kondisi |
|---|---|
| Before | Audit dataset NEU dan MVTec, dataset discovery FiftyOne, pemeriksaan anotasi, hash dan split, eksperimen baseline. |
| Now | Notebook + report penelitian tersedia; model `.pt`, `.onnx` dan 15 `.joblib` tersimpan; FastAPI + React dapat menjalankan dua pipeline. |
| Next | Automated API tests/CI, kalibrasi threshold di data validasi, UI polish, pin dependency, pengujian dunia nyata, keamanan/izin komersial. |

## Dataset asli dan audit

**NEU-DET:** sumber Kaggle terlapor `/kaggle/input/datasets/wafabila/neu-det`; enam kelas `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`. Split tercatat **1.259 train, 270 val, 270 test**, satu exact duplicate disisihkan, grouping berdasarkan hash + keluarga nama file. Risiko near-duplicate masih ada. Data tidak digabung dengan MVTec.

**MVTec:** sumber `/kaggle/input/datasets/wafabila/mvtec-ad-dataset`; format ekspor FiftyOne memakai `samples.json` dan `data/data_*`. Total test penelitian **1.725 gambar di 15 kategori**, bank hanya dari `train/good`; exact train-test overlap per kategori nol pada audit dan tidak ada mask cacat yang hilang pada hasil audit. Lihat `reports/mvtec_integrity.json` dan `reports/mvtec_memory_audit.json`.

## Metrik hasil eksperimen (bukan angka live)

| Pipeline | Metrik | Nilai |
|---|---|---:|
| NEU-DET | Test precision | 71,89% |
| NEU-DET | Test recall | 73,65% |
| NEU-DET | Test mAP@50 | 78,52% |
| NEU-DET | Test mAP@50–95 | 46,03% |
| MVTec AD | Macro image AUROC 15 kategori | 81,17% |
| MVTec AD | Macro **coarse pixel** AUROC 16×16 | 88,82% |

Source-of-truth: `reports/final_summary.json`, `neu_heldout_metrics.json`, `mvtec_heldout_metrics.csv`, `mvtec_summary.json`. Jangan menulis AUROC sebagai “akurasi”, atau menyebut coarse pixel metric sebagai segmentasi tingkat piksel penuh. Contoh test botol: skor normal `0.7583` dan `broken_large` `0.9847`; itu hanya **sanity check satu pasangan**, bukan threshold.

## Instalasi lokal Windows

Terminal 1:

```powershell
cd "D:\porto\100 day\day 22\DefectSpot\backend"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Terminal 2:

```powershell
cd "D:\porto\100 day\day 22\DefectSpot\frontend"
npm ci
npm run build
npm run dev
```

Swagger: `http://127.0.0.1:8000/docs`; UI: `http://127.0.0.1:5173`. Path Windows di atas contoh lokasi kerja; kode runtime tidak bergantung pada nama drive.

## Cara menggunakan

- Tab **NEU-DET**: upload gambar baja, threshold default 0,25, klik Run Inspection, baca bounding boxes dan confidence. Tidak ada box bukan bukti baja bebas cacat.
- Tab **MVTec AD**: pilih kategori produk sesuai gambar (misalnya `bottle`), upload, jalankan, baca skor jarak anomali dan overlay. Semakin tinggi **biasanya** semakin tidak serupa dengan normal untuk kategori tersebut; ini **bukan peluang/probabilitas cacat**.
- Skor MVTec tidak boleh dibandingkan antar kategori; warna heatmap diskalakan per gambar sehingga tidak boleh langsung dibandingkan antar gambar.

## Pengujian dan troubleshooting

- `python -m pytest tests/test_mvtec_features.py -q` dari folder backend (dalam virtualenv).
- `npm run build` dan `npm run typecheck` dari frontend.
- Cek `GET /api/v1/health` dan `GET /api/v1/mvtec/categories` sebelum inference.
- **API Offline:** jalankan backend port 8000 / cek Vite proxy.
- **503:** model hilang, class contract mismatch, atau joblib tidak kompatibel; lihat log backend.
- **415/413/422:** file tidak didukung, terlalu besar, terlalu banyak piksel, atau kategori salah.
- **MVTec 501:** backend lama masih berjalan; update/restart app dan periksa route tidak ganda.
- **Build TS error `ImportMeta.env`:** cek `frontend/src/vite-env.d.ts`.

## Keterbatasan, klaim, dan deployment

Belum ada calibration threshold, keputusan QA terotomasi, SLA line-speed, sertifikasi industri, evaluasi domain pabrik, atau keamanan deployment produksi. Gunakan *human-in-the-loop*. Deployment publik perlu auth, HTTPS, pengamanan upload dan model, logging/monitoring, serta audit lisensi.

**Lisensi:** root `LICENSE` MIT untuk kode asli; MVTec AD CC BY-NC-SA 4.0 (non-komersial), lisensi NEU perlu diverifikasi, dan pretrained model/tool pihak ketiga punya ketentuan sendiri. MIT tidak otomatis membuat semua dataset/weights bebas dipakai komersial.

## FAQ singkat

**Perlu training ulang?** Tidak untuk menjalankan demo saat ini; model sudah tersedia. Untuk mengganti metodologi atau dataset, eksperimen harus didokumentasikan sebagai run baru.

**MVTec memakai YOLO?** Tidak. Bank normal-only handcrafted + nearest neighbor.

**Skor 0,9847 artinya 98,47% cacat?** Tidak. Itu jarak fitur mentah belum dikalibrasi.

**Bisa langsung dipakai di pabrik?** Belum; butuh validasi, prosedur inspeksi, keamanan dan izin penggunaan data/model.

**Di mana struktur repo, roadmap, FAQ lengkap dan source-of-truth?** Di [`README.md`](../../README.md) dan [`AGENTS.md`](../../AGENTS.md).
