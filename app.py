"""AERIS operator research interface. Run: python -m streamlit run app.py"""
from pathlib import Path
import json,sys,time,uuid
import numpy as np
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from aeris.modeling import load_hybrid, CLASS_NAMES, sha256_file
from aeris.workflow import diagnose, canonical_window, feedback_update, restore_packaged_forests
restore_packaged_forests(ROOT)
st.set_page_config(page_title='AERIS · AHU diagnosis',page_icon='◉',layout='wide')
st.markdown('''<style>.stApp{background:#f5f7fb} h1,h2,h3{letter-spacing:-.025em} [data-testid="stMetric"]{background:white;padding:18px;border-radius:12px;border:1px solid #e3e8ef}</style>''',unsafe_allow_html=True)
st.title('AERIS · AHU diagnosis lab')
st.caption('Hybrid Random Forest + temporal neural networks · Research decision support · No automatic equipment actuation')
@st.cache_resource
def bundle(name, identity):
 m=load_hybrid(ROOT/'data/aeris_models'/name);m.forest.n_jobs=1;return m
@st.cache_data
def data(role):
 w=ROOT/'data/aeris_windows_v1';p={k:v for k,v in np.load(w/f'{role}.npz',allow_pickle=False).items()}
 meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id').loc[p['window_id']].reset_index();return p,meta
available=[p.name for p in (ROOT/'data/aeris_models').glob('m4_*') if (p/'READY.json').exists()]
if not available:st.info('Training is running; complete bundles appear when checksum and reload checks pass.');st.stop()
selection=ROOT/'milestone4/evidence/component_selection.json'
default=json.loads(selection.read_text())['selected_initial_bundle'] if selection.exists() else 'm4_v0'
with st.sidebar:
 st.header('Model workspace')
 name=st.selectbox('Trained bundle',sorted(available),index=sorted(available).index(default) if default in available else 0)
 st.caption('V0: final step · V1: attention · V2: combined. Adaptation and feedback retain the architecture identity.')
 st.warning('Three shared-weather source groups. Target data are retrospective and exploratory, not a sealed holdout.')
model=bundle(name,sha256_file(ROOT/'data/aeris_models'/name/'hybrid.json'))
view,comparison,learning,evidence=st.tabs(['Diagnosis','Experiments','Adaptation & feedback','Evidence'])
with view:
 source=st.radio('Input',['Recorded window','Upload canonical CSV'],horizontal=True)
 raw=None;times=None;wid=None
 if source=='Recorded window':
  role=st.selectbox('Recording pool',['source_validation','target_test'])
  pack,meta=data(role);ix=st.number_input('Window index',min_value=0,max_value=len(meta)-1,value=0,step=1)
  row=meta.iloc[ix];wid=str(row.window_id);raw=canonical_window(pack,ix,model.scaler);times=np.arange(15)*60+row.start_time_s
  setting=row.season if role.startswith('source') else row.dataset
  st.caption(f'{wid} · {setting} · {row.condition_group}')
 else:
  st.caption('15 consecutive one-minute rows. Required: time_s, then 10 canonical columns. Temperatures in °C; commands 0–1. No imputation or silent unit inference.')
  features=model.protocol['core_features'];template=pd.DataFrame(columns=['time_s']+features)
  st.download_button('Download CSV template',template.to_csv(index=False),'aeris_input_template.csv','text/csv')
  file=st.file_uploader('Canonical sensor CSV',type='csv');setting=st.selectbox('AHU setting',['CoolingSeason','HeatingSeason','ShoulderSeason','SZVAV','SZCAV','Unknown'])
  if file:
   try:
    frame=pd.read_csv(file)
    if len(frame)!=15 or not set(['time_s']+features).issubset(frame.columns):raise ValueError('Expected exactly 15 rows and all required columns.')
    raw=frame[features].apply(pd.to_numeric,errors='raise').to_numpy('float32');times=pd.to_numeric(frame.time_s,errors='raise').to_numpy();wid='upload-'+uuid.uuid4().hex[:12]
   except (ValueError,KeyError) as exc:st.error(str(exc))
 if raw is not None:
  started=time.perf_counter();result=diagnose(model,raw,setting=setting,timestamps=times,recorded_source=(source=='Recorded window' and role.startswith('source')));elapsed=time.perf_counter()-started
  st.session_state['last_result']=result
  if result['status']=='blocked':st.error('Input blocked: '+' '.join(result['reasons']))
  else:
   a,b,c=st.columns(3);a.metric('Suggested condition',result['class_name'].replace('_',' ').title());b.metric('Model confidence',f"{result['confidence']:.1%}");c.metric('Request compute',f'{elapsed*1000:.1f} ms')
   if result['reasons']:st.warning('Review required: '+' '.join(result['reasons']))
   else:st.info('Research prediction available. A technician must confirm the physical condition before action.')
   st.bar_chart(pd.DataFrame({'Hybrid':result['probabilities'],'Forest':result['forest_probabilities'],'Temporal':result['temporal_probabilities']},index=CLASS_NAMES))
   st.line_chart(pd.DataFrame(raw[:,:4],columns=features[:4] if source!='Recorded window' else model.protocol['core_features'][:4]))
   st.caption('Global forest feature importance and temporal attention describe model behavior; neither proves a physical cause.')
   with st.expander('Input rows and audit'):st.dataframe(pd.DataFrame(raw,columns=model.protocol['core_features']));st.json(result)
   st.download_button('Export diagnosis audit',json.dumps({'window_id':wid,'bundle':name,'setting':setting,'result':result},indent=2),'aeris_diagnosis.json','application/json')
   with st.form('operator_feedback'):
    st.subheader('Technician review queue')
    note=st.text_area('Observation / inspection note');label=st.selectbox('Inspection label',['Unverified']+CLASS_NAMES)
    verified=st.checkbox('I have independently verified this label by inspection (not by confidence alone).')
    if st.form_submit_button('Save review record'):
     if verified and label=='Unverified':st.error('Select an inspected label or leave the record unverified.')
     else:
      folder=ROOT/'data/operator_feedback';folder.mkdir(parents=True,exist_ok=True)
      record={'window_id':wid,'bundle':name,'note':note,'class_name':label,'verified':verified,'training_eligible':False,'reason':'Operator records require dataset alignment and a controlled reviewed update; saving never changes weights.'}
      (folder/f'{uuid.uuid4().hex}.json').write_text(json.dumps(record,indent=2));st.success('Review saved. Model weights are unchanged.')
with comparison:
 path=ROOT/'milestone4/evidence/results.csv'
 if path.exists():st.dataframe(pd.read_csv(path),hide_index=True,width='stretch')
 if selection.exists():st.json(json.loads(selection.read_text()))
 st.caption('Controlled initial runs: fresh RF200 + fresh temporal NN50 epochs; seed42, fixed .5/.5 fusion, identical packs. All scores describe these recordings.')
 errors=ROOT/'milestone4/evidence/error_examples.csv'
 if errors.exists():st.subheader('Real model failures');st.dataframe(pd.read_csv(errors),hide_index=True)
with learning:
 st.subheader('Cross-setting adaptation and verified feedback')
 st.write('Unlabeled target data pass fixed confidence, agreement, support, stability and diversity gates. Source replay and source retention checks control updates. A skipped or rejected update retains its parent.')
 reports=[]
 for p in (ROOT/'data/aeris_models').glob('m4_*'):
  for f in ['selection_summary.json','feedback_report.json']:
   if (p/f).exists():reports.append({'bundle':p.name,'report':f,**json.loads((p/f).read_text())})
 for report in reports:
  with st.expander(report['bundle']+' · '+report['report']):st.json(report)
 st.info('Published reference labels are used only in the explicitly simulated technician experiment. Unverified operator notes are never used as labels.')
 fresh_name=st.selectbox('Architecture for controlled feedback replay',['m4_v0','m4_v1','m4_v2'])
 if st.button('Run controlled reference-feedback experiment',disabled=(ROOT/'data/aeris_models'/f'{fresh_name}_feedback').exists()):
  with st.spinner('Training candidate and checking source retention…'):feedback_update(ROOT,fresh_name)
  st.success('Feedback experiment complete; inspect its acceptance/rollback report.');st.rerun()
with evidence:
 st.subheader('Reproducibility and scope')
 for f in ['readiness/audit.json','readiness/scope.json','validation_report.json','pipeline_status.json']:
  p=ROOT/'milestone4/evidence'/f
  if p.exists():
   with st.expander(f):st.json(json.loads(p.read_text()))
 st.link_button('Original project repository','https://github.com/mharsha100/aeris-ahu-fault-diagnosis')
