export type Detection = {class_id:number; class_name:string;confidence:number;bbox:{x1:number;y1:number;x2:number;y2:number}};
export type Prediction = {pipeline:'neu_det';model:string;width:number;height:number;detections:Detection[];detection_count:number;inference_ms:number;annotated_image_base64:string|null;quality_decision:string;note:string};
export type ModelInfo = {detection:{task:string;classes:string[];checkpoint:string};anomaly:{task:string;categories:string[];status:string};warning:string};
export type HealthInfo={status:string;neu_checkpoint_present:boolean;mvtec_banks_present:number;mvtec_runtime:string};
const BASE=(import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/,'');
export async function readJson<T>(url:string, options?:RequestInit):Promise<T>{
 const res=await fetch(`${BASE}${url}`,options);
 if(!res.ok){const body=await res.json().catch(()=>null);const detail=body?.detail;throw new Error(typeof detail==='string'?detail:detail?JSON.stringify(detail):`HTTP ${res.status}`)}
 return res.json() as Promise<T>;
}
export const health=()=>readJson<HealthInfo>('/health');
export const models=()=>readJson<ModelInfo>('/models');
export async function predictNeu(file:File, confidence:number):Promise<Prediction>{
 const form=new FormData();form.append('file',file);
 return readJson<Prediction>(`/predict/neu?confidence=${confidence.toFixed(2)}&include_annotated=true`,{method:'POST',body:form});
}
export function saveBlob(data:Blob,name:string){const url=URL.createObjectURL(data);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
