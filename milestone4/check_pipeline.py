"""Meaningful guardrail, isolation, architecture and full UI checks."""
from pathlib import Path
import sys,json,copy,hashlib,time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.modeling import load_hybrid,sha256_file,create_temporal
from aeris.workflow import diagnose,canonical_window,choose_feedback
from aeris.training import write_json

def run():
 torch.set_num_threads(2);checks=[];w=ROOT/'data/aeris_windows_v1';pack=dict(np.load(w/'source_validation.npz',allow_pickle=False));m=load_hybrid(ROOT/'data/aeris_models/m4_v2');raw=canonical_window(pack,0,m.scaler);times=np.arange(15)*60
 def check(name,passed):
  if not passed:raise AssertionError(name)
  checks.append({'check':name,'status':'passed'})
 for name,modified in [('missing',np.full((15,10),np.nan)),('fan_off',raw.copy()),('unoccupied',raw.copy()),('command_range',raw.copy())]:
  if name=='fan_off':modified[:,4]=0
  if name=='unoccupied':modified[:,9]=0
  if name=='command_range':modified[:,6]=1.2
  check('Block '+name,diagnose(m,modified,setting='CoolingSeason',timestamps=times)['status']=='blocked')
 check('Block timestamp gap',diagnose(m,raw,setting='CoolingSeason',timestamps=times*2)['status']=='blocked')
 original=m.predict
 def confident(seq,tab):return {'class_id':np.array([2]),'confidence':np.array([.999]),'branch_disagreement':np.array([False]),**{k:np.array([[.0003,.0003,.999,.0004]]) for k in ['probabilities','forest_probabilities','temporal_probabilities']}}
 m.predict=confident
 check('High supplied confidence cannot override exploratory target',diagnose(m,raw,setting='SZVAV',timestamps=times)['status']=='review_required')
 check('High supplied confidence cannot override class/season scope',diagnose(m,raw,setting='HeatingSeason',timestamps=times)['status']=='review_required')
 check('Unverified uploaded source context always requires review',diagnose(m,raw,setting='CoolingSeason',timestamps=times)['status']=='review_required')
 m.predict=original
 for version,kind in [('v0','causal_tcn_last'),('v1','causal_tcn_attention'),('v2','causal_tcn_combined')]:
  model=load_hybrid(ROOT/'data/aeris_models'/f'm4_{version}');run=json.loads((ROOT/'data/aeris_models'/f'm4_{version}/training_run.json').read_text())
  check(version+' full50 fresh epochs',run['completed_temporal_epochs']==50 and not run['parent_weights_loaded'] and not run['early_stopping_enabled'])
  check(version+' correct architecture/fusion',model.configuration['temporal_kind']==kind and model.configuration['forest_weight']==.5)
  x=torch.randn(2,15,10);altered=x.clone();altered[:,8:,:]+=7
  with torch.no_grad():a=model.temporal.blocks(x.transpose(1,2));b=model.temporal.blocks(altered.transpose(1,2))
  check(version+' encoder causal prefix',torch.allclose(a[:,:,:8],b[:,:,:8],atol=1e-6))
  for suffix in ['_cross_adapted','_feedback']:
   stage=ROOT/'data/aeris_models'/f'm4_{version}{suffix}';restored=load_hybrid(stage)
   check(version+suffix+' architecture retained',restored.configuration['temporal_kind']==kind)
   check(version+suffix+' no test fitting',not restored.configuration['test_data_used'])
 feedback=load_hybrid(ROOT/'data/aeris_models/m4_v2_feedback');feedback.predict=confident
 check('Source guard does not authorize feedback candidate promotion',diagnose(feedback,raw,setting='CoolingSeason',timestamps=times,recorded_source=True)['status']=='review_required')
 # Original Git LF blobs must be identical to preserved m3-v1.
 import subprocess
 for p in (ROOT/'milestone3').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:
   expected=subprocess.check_output(['git','show','ab54b636082bb8d4ad7142483f663be5d60d8a0d:'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
   check('Preserved '+p.relative_to(ROOT).as_posix(),p.read_bytes()==expected)
 # A real Streamlit run checks widget logic and recorded-window inference.
 from streamlit.testing.v1 import AppTest
 app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
 check('Streamlit starts without exceptions',not list(app.exception))
 check('Streamlit renders actual diagnosis',len(app.metric)>=3 and app.metric[1].value.endswith('%'))
 app.radio[0].set_value('Upload canonical CSV').run();check('CSV upload screen renders',not list(app.exception))
 app.radio[0].set_value('Recorded window').run();next(x for x in app.selectbox if x.label=='Recording pool').set_value('target_test').run()
 check('Target request requires review in UI',not list(app.exception) and any('Review required' in x.value for x in app.warning))
 app.text_area[0].set_value('QA fixture: interface test, not an observed inspection.').run()
 app.checkbox[0].set_value(True).run()
 next(x for x in app.button if x.label=='Save review record').click().run()
 check('Unverified label cannot be saved as verified',not list(app.exception) and any('Select an inspected label' in x.value for x in app.error))
 app.checkbox[0].set_value(False).run()
 next(x for x in app.button if x.label=='Save review record').click().run()
 check('Unverified observation saved without training',not list(app.exception) and any('weights are unchanged' in x.value for x in app.success))
 notes=list((ROOT/'data/operator_feedback').glob('*.json'))
 check('Operator note excluded from learning',any(json.loads(p.read_text()).get('training_eligible') is False for p in notes))
 report={'status':'passed','checks':checks,'passed_checks':len(checks),'fixtures_are_real_model_accuracy_evidence':False,
 'real_training_and_evaluation_evidence':'architecture_training.json,results.csv,error_examples.csv',
 'ui':'Streamlit actual widget/runtime tests passed; hosted browser checked separately'}
 write_json(ROOT/'milestone4/evidence/validation_report.json',report);print(f'{len(checks)} checks passed',flush=True)
if __name__=='__main__':run()
