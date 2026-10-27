"""Inspect trusted local Day 22 artifacts, without executing inference.
Warning: joblib deserialization can execute arbitrary code: ONLY run on your own trusted exports.
"""
from pathlib import Path
from collections.abc import Mapping
import json

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut',
            'pill','screw','tile','toothbrush','transistor','wood','zipper')

def describe(v):
    return {'type':type(v).__module__+'.'+type(v).__name__,
            'shape':list(v.shape) if hasattr(v,'shape') else None,
            'dtype':str(v.dtype) if hasattr(v,'dtype') else None,
            'length':len(v) if hasattr(v,'__len__') else None}

def main():
    import joblib
    output={'banks':{},'detection':{}}
    for category in CATEGORIES:
        file=ROOT/'models'/f'mvtec_patch_{category}.joblib'
        if not file.is_file():
            output['banks'][category]={'error':'missing'};continue
        obj=joblib.load(file) # trusted files ONLY
        info=describe(obj)
        if isinstance(obj,Mapping):
            info['keys']={str(k):describe(v) for k,v in obj.items()}
        output['banks'][category]=info
    try:
        from ultralytics import YOLO
        detector=YOLO(str(ROOT/'models'/'neu_best.pt'))
        output['detection']={'task':detector.task,'names':detector.names}
    except Exception as exc:
        output['detection']={'error':str(exc)}
    target=ROOT/'reports'/'model_contract_inspection.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps(output,indent=2,ensure_ascii=False,default=str))
    print(f'Wrote {target}')
if __name__=='__main__':main()
