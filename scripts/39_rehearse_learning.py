"""Explicit TEST-cohort rehearsal; published labels simulate technician answers."""
from pathlib import Path
import sys,time,json,uuid,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from aeris.modeling import load_hybrid,CLASS_NAMES
from aeris.workflow import choose_feedback
OUT=ROOT/'experiments/adaptive_lifecycle_20261117';OUT.mkdir(parents=True,exist_ok=True)
API='http://127.0.0.1:8011'
def api(path,body=None):
 req=urllib.request.Request(API+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
def wait(stage):
 start=time.time()
 while time.time()-start<7200:
  status=api('/learning');jobs=[x for x in status['jobs'] if x['cohort']=='test' and x['stage']==stage]
  (OUT/'rehearsal_status.json').write_text(json.dumps(status,indent=2))
  if jobs and jobs[0]['status']!='training':
   print(stage,jobs[0]['status'],jobs[0]['error'],flush=True);return jobs[0]
  time.sleep(10)
 raise TimeoutError('Background learning did not finish')
def main():
 evidence={'scope':'Test-cohort rehearsal only; reviewed labels are published references, not real inspections. Genuine registry must remain unchanged.','before':api('/learning'),'predictions':[],'reviews':[]}
 # Frozen from the source-only model using confidence and input checks, no labels.
 for i in [27,28,29,30,31,32,33,36]:
  r=api('/recorded',dict(request_id='lifecycle-v1-demo-'+str(i),cohort='test',role='target_adapt',index=i));evidence['predictions'].append(r)
 print('Submitted 8 confident adaptation windows',flush=True)
 v1=wait('v1');evidence['v1_job']=v1
 (OUT/'rehearsal.json').write_text(json.dumps(evidence,indent=2))
 # Freeze low-confidence/disagreement selection before revealing any reference labels.
 w=ROOT/'data/aeris_windows_v1';p=dict(np.load(w/'target_feedback.npz'));meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id').loc[p['window_id']].reset_index()
 m=load_hybrid(ROOT/'data/aeris_models/boosted_v2_s42');m.forest.n_jobs=2;pred=m.predict(p['X_seq'],p['X_tab']);ix=choose_feedback(p,meta,pred,limit=20)
 (OUT/'review_selection.json').write_text(json.dumps(dict(indices=ix.tolist(),window_ids=p['window_id'][ix].tolist(),selection='Disagreement/low confidence; max 3 per condition, non-overlapping; labels unopened during this selection.'),indent=2))
 reference=pd.read_csv(w/'evaluation/target_window_reference.csv.gz').set_index('window_id')
 for i in ix:
  r=api('/recorded',dict(request_id='lifecycle-v2-demo-'+str(i),cohort='test',role='target_feedback',index=int(i)));label=CLASS_NAMES[int(reference.loc[p['window_id'][i],'class_id'])]
  f=api('/feedback',dict(prediction_id=r['prediction_id'],feedback=dict(corrected_class=label,inspector='Automated demonstration (not a human technician)',provenance='Published target_feedback reference for '+str(p['window_id'][i])+'; simulated technician label, TEST cohort only.',note='Reproducible pipeline demonstration; no independent inspection claimed.')))
  evidence['reviews'].append(dict(prediction=r,feedback=f))
 print('Submitted',len(ix),'explicitly simulated technician labels',flush=True)
 evidence['v2_job']=wait('v2');evidence['after']=api('/learning')
 assert evidence['after']['active']['genuine']==evidence['before']['active']['genuine']
 (OUT/'rehearsal.json').write_text(json.dumps(evidence,indent=2));print('REHEARSAL COMPLETE',flush=True)
if __name__=='__main__':main()
