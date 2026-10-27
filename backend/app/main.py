"""DefectSpot API. NEU-DET inference is implemented; MVTec bank runtime pending contract verification."""
from __future__ import annotations
import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'neu_best.pt'
MVTEC_CATEGORIES = ('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut',
                    'pill','screw','tile','toothbrush','transistor','wood','zipper')
NEU_CLASSES = ('crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches')

app = FastAPI(title='DefectSpot Inference API',version='0.1.0',description='Two independent research tasks; NEU inference implemented, MVTec contract under verification.')
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET','POST'],allow_headers=['*'])
_lock = Lock()

@lru_cache(maxsize=1)
def get_detector():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f'NEU checkpoint missing: {MODEL_PATH}')
    from ultralytics import YOLO
    detector = YOLO(str(MODEL_PATH))
    names = {int(k):str(v) for k,v in detector.names.items()}
    if detector.task != 'detect' or names != dict(enumerate(NEU_CLASSES)):
        raise RuntimeError(f'Checkpoint class contract mismatch: {names!r}, task={detector.task!r}')
    return detector

async def read_image(file: UploadFile)->Image.Image:
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415,'Only JPEG, PNG, and WebP supported')
    data = await file.read(10*1024*1024+1)
    if not data:
        raise HTTPException(400,'Empty upload')
    if len(data)>10*1024*1024:
        raise HTTPException(413,'Image exceeds 10 MB')
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width*source.height > 16_000_000:
                raise HTTPException(413,'Image too large (pixel limit)')
            source.load()
            return ImageOps.exif_transpose(source).convert('RGB')
    except HTTPException:
        raise
    except (UnidentifiedImageError,OSError,ValueError) as exc:
        raise HTTPException(422,'Invalid image') from exc

@app.get('/api/v1/health')
def health():
    return {'status':'ok','neu_checkpoint_present':MODEL_PATH.is_file(),
            'mvtec_banks_present':sum((ROOT/'models'/f'mvtec_patch_{name}.joblib').is_file() for name in MVTEC_CATEGORIES),
            'mvtec_runtime':'requires_feature_contract_verification'}

@app.get('/api/v1/models')
def models():
    return {'detection':{'task':'object_detection','classes':list(NEU_CLASSES),'checkpoint':'models/neu_best.pt'},
            'anomaly':{'task':'normal_only_patch_anomaly','categories':list(MVTEC_CATEGORIES),
                       'status':'artifact_present_runtime_not_yet_verified'},
            'warning':'Metrics from independent tasks must not be compared directly.'}

@app.post('/api/v1/predict/neu')
async def predict_neu(file:UploadFile=File(...),confidence:float=Query(0.25,ge=0.01,le=1),include_annotated:bool=False):
    image = await read_image(file)
    try:
        model=get_detector()
        import numpy as np
        t=time.perf_counter()
        with _lock:
            result=model.predict(source=np.asarray(image),conf=confidence,verbose=False,imgsz=640)[0]
        elapsed=round((time.perf_counter()-t)*1000,2)
        rows=[]
        for b in (result.boxes if result.boxes is not None else []):
            idx=int(b.cls.item())
            if idx<0 or idx>=len(NEU_CLASSES):
                raise ValueError('Unexpected class ID')
            xyxy=[round(float(v),2) for v in b.xyxy[0].tolist()]
            rows.append({'class_id':idx,'class_name':NEU_CLASSES[idx],
                         'confidence':round(float(b.conf.item()),6),'bbox':dict(zip(('x1','y1','x2','y2'),xyxy))})
        annotated=None
        if include_annotated:
            from PIL import ImageDraw
            im=image.copy();draw=ImageDraw.Draw(im)
            for d in rows:
                bb=d['bbox'];draw.rectangle(tuple(bb[k] for k in ('x1','y1','x2','y2')),outline='#24c4a6',width=3)
                draw.text((bb['x1'],max(0,bb['y1']-15)),f"{d['class_name']} {d['confidence']:.2f}",fill='#24c4a6')
            buf=BytesIO();im.save(buf,'JPEG',quality=85)
            annotated=base64.b64encode(buf.getvalue()).decode('ascii')
        return {'pipeline':'neu_det','model':'neu_best.pt','width':image.width,'height':image.height,
                'detections':rows,'detection_count':len(rows),'inference_ms':elapsed,
                'annotated_image_base64':annotated,'quality_decision':'not_assessed',
                'note':'Detections are visual observations, not validated manufacturing acceptance decisions.'}
    except RuntimeError as exc:
        raise HTTPException(503,str(exc)) from exc

@app.post('/api/v1/predict/mvtec')
async def mvtec_unavailable(category:str=Query(...)):
    if category not in MVTEC_CATEGORIES:
        raise HTTPException(422,'Unknown MVTec category')
    raise HTTPException(501,'MVTec inference requires verified joblib object schema and original 128px/16x16 descriptor preprocessing; never substitute an approximate feature extractor.')
