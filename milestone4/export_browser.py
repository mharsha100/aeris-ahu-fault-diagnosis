"""Export actual fitted trees/weights and real cases for local browser inference."""
from pathlib import Path
import gzip,json,sys,subprocess,shutil,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.modeling import load_hybrid,sha256_file
from aeris.workflow import canonical_window,restore_packaged_forests
from aeris.training import write_json

def gzjson(path,data):
 with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as f:json.dump(data,f,separators=(',',':'))
def read(p):return json.loads(p.read_text())
def run():
 restore_packaged_forests(ROOT)
 out=ROOT/'dist';assets=out/'models';assets.mkdir(parents=True,exist_ok=True);w=ROOT/'data/aeris_windows_v1';models={};fixtures=[]
 for n in ['m4_v0','m4_v1','m4_v2']:
  for suffix in ['','_cross_adapted','_feedback']:
   name=n+suffix;b=ROOT/'data/aeris_models'/name;m=load_hybrid(b);m.forest.n_jobs=1
   fh=sha256_file(b/'forest.joblib')[:20];th=sha256_file(b/'temporal_state.pt')[:20]
   fp=assets/f'{fh}.json.gz';tp=assets/f'{th}.json.gz'
   if not fp.exists():
    trees=[]
    for estimator in m.forest.estimators_:
     t=estimator.tree_;trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,:].tolist()})
    gzjson(fp,trees)
   if not tp.exists():gzjson(tp,{k:v.tolist() for k,v in m.temporal.state_dict().items()})
   models[name]={'config':m.configuration,'scaler':m.scaler,'forest_url':'models/'+fp.name,'temporal_url':'models/'+tp.name,'hybrid_sha256':sha256_file(b/'hybrid.json')}
   pack=dict(np.load(w/'source_validation.npz',allow_pickle=False));ix=np.random.default_rng(42).choice(len(pack['y']),100,replace=False)
   pred=m.predict(pack['X_seq'][ix],pack['X_tab'][ix])
   raws=np.stack([canonical_window(pack,int(i),m.scaler) for i in ix]);raw_pred=m.predict_raw(raws)
   fixtures.append({'model':name,'raw':raws.tolist(),'raw_probabilities':raw_pred['probabilities'].tolist(),'seq':pack['X_seq'][ix].tolist(),'tab':pack['X_tab'][ix].tolist(),'probabilities':pred['probabilities'].tolist()})
   print('Exported actual model',name,flush=True)
 meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id');ref=pd.read_csv(w/'evaluation/target_window_reference.csv.gz').set_index('window_id');cases=[]
 # Real source windows plus later target windows, with all four classes represented.
 for role in ['source_validation','target_test']:
  pack=dict(np.load(w/f'{role}.npz',allow_pickle=False));y=pack['y'] if 'y'in pack else ref.loc[pack['window_id'],'class_id'].to_numpy()
  ids=[]
  for k in range(4):
   c=np.flatnonzero(y==k);ids.extend(c[np.linspace(0,len(c)-1,min(20,len(c)),dtype=int)].tolist())
  for ix in ids:
   wid=str(pack['window_id'][ix]);row=meta.loc[wid];r=canonical_window(pack,ix,m.scaler)
   cases.append({'id':wid,'pool':'Source' if role.startswith('source') else 'Target','setting':row.season if role.startswith('source') else row.dataset,
    'group':row.condition_group,'raw':r.tolist(),'time_s':(row.start_time_s+np.arange(15)*60).tolist(),'truth':int(y[ix])})
 write_json(out/'cases.json',cases)
 ev=ROOT/'milestone4/evidence';learning={}
 for n in ['m4_v0','m4_v1','m4_v2']:
  b=ROOT/'data/aeris_models';learning[n]={'initial':read(b/n/'training_run.json'),'adaptation':read(b/f'{n}_cross_adapted/selection_summary.json'),
    'adaptation_status':read(b/f'{n}_cross_adapted/hybrid.json')['adaptation_status'],'feedback':read(b/f'{n}_feedback/feedback_report.json')}
 def records(f):return json.loads(pd.read_csv(ev/f).to_json(orient='records'))
 write_json(out/'evidence.json',{'selection':read(ev/'component_selection.json'),'results':records('results.csv'),'thresholds':records('review_threshold_experiment.csv'),
   'errors':records('error_review_20.csv'),'validation':read(ev/'validation_report.json'),'learning':learning})
 write_json(out/'manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'models':models,
  'features':m.protocol['core_features'],'cases_url':'cases.json','evidence_url':'evidence.json','artifacts':[
   {'label':'Complete runnable project + trained models','url':'downloads/aeris_m4_complete.zip'},
   {'label':'Experiment evidence','url':'downloads/aeris_m4_evidence.zip'},
   {'label':'Verification report','url':'validation_report.json'}]})
 shutil.copyfile(ev/'validation_report.json',out/'validation_report.json')
 write_json(ROOT/'data/browser_parity_fixtures.json',fixtures)
if __name__=='__main__':run()
