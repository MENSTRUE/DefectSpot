import { useEffect, useState } from 'react';
import './MvtecInspection.css';

type Result = {
  pipeline: string;
  category: string;
  model: string;
  image: { width: number; height: number; filename: string };
  anomaly_score_raw: number;
  heatmap_grid: [number, number];
  patch_distances: number[][];
  inference_ms: number;
  decision: string;
  note: string;
  overlay_jpeg_base64: string | null;
};
const CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'];
const BASE = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '');

export default function MvtecInspection() {
  const [category, setCategory] = useState('bottle');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const onFile = (picked?: File) => {
    setResult(null); setError('');
    if (!picked) return setFile(null);
    if (!['image/jpeg','image/png','image/webp'].includes(picked.type) || picked.size > 10 * 1024 * 1024) {
      setFile(null); setError('Gunakan JPG, PNG, atau WebP maksimal 10 MB.'); return;
    }
    setFile(picked);
  };
  async function inspect() {
    if (!file) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const form = new FormData(); form.append('file', file);
      const params = new URLSearchParams({category, include_overlay: 'true'});
      const response = await fetch(`${BASE}/predict/mvtec?${params}`, {method:'POST',body:form});
      if (!response.ok) {
        let details = `HTTP ${response.status}`;
        try { const payload = await response.json(); details = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail); } catch {}
        throw new Error(details);
      }
      setResult(await response.json() as Result);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }
  function exportJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify({...result, overlay_jpeg_base64: undefined},null,2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a'); anchor.href=url;anchor.download=`defectspot-mvtec-${category}.json`;anchor.click();
    URL.revokeObjectURL(url);
  }
  return <section className="mvtec-panel" aria-label="MVTec anomaly inspection">
    <div className="mvtec-head"><div><h2>MVTec AD Â· Visual anomaly</h2><p>Normal-only patch nearest-neighbor Â· 15 kategori Â· heatmap 16Ã—16</p></div><span className="mvtec-status">Research prototype</span></div>
    <div className="mvtec-form">
      <label>Kategori produk <select value={category} onChange={e=>{setCategory(e.target.value);setResult(null);}}>{CATEGORIES.map(c=><option key={c} value={c}>{c}</option>)}</select></label>
      <label>Gambar produk <input type="file" accept="image/png,image/jpeg,image/webp" onChange={e=>onFile(e.target.files?.[0])}/></label>
      <button disabled={!file || loading} onClick={inspect}>{loading ? 'Analyzing...' : 'Run anomaly inspection'}</button>
    </div>
    {error && <p className="mvtec-error" role="alert">{error}</p>}
    <div className="mvtec-images">
      {preview && <figure><img src={preview} alt="Uploaded product"/><figcaption>Original</figcaption></figure>}
      {result?.overlay_jpeg_base64 && <figure><img src={`data:image/jpeg;base64,${result.overlay_jpeg_base64}`} alt="Coarse anomaly heatmap overlay"/><figcaption>Coarse patch heatmap overlay Â· display scaling per image</figcaption></figure>}
    </div>
    {result && <div className="mvtec-result">
      <div><span>Raw anomaly distance</span><strong>{result.anomaly_score_raw.toFixed(4)}</strong></div>
      <div><span>Inference</span><strong>{result.inference_ms.toFixed(1)} ms</strong></div>
      <div><span>Decision</span><strong>Not assessed</strong></div>
      <p>{result.note}</p><p>Nilai skor tidak boleh dibandingkan langsung antar-kategori atau diperlakukan sebagai probabilitas cacat.</p>
      <button onClick={exportJson}>Export JSON (without embedded image)</button>
    </div>}
  </section>;
}

