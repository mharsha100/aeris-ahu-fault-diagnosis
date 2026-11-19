"""AERIS operator interface backed by the durable local inference service."""
from pathlib import Path
import os,json,uuid,time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parent
API=os.environ.get('AERIS_API','http://127.0.0.1:8001')
CLASSES=['normal','damper_fault','cooling_valve_fault','heating_valve_fault']
st.set_page_config(page_title='AERIS · AHU diagnosis',page_icon='◉',layout='wide')
st.title('AERIS · AHU diagnosis lab')
st.caption('Random Forest + temporal neural network · Research decision support · No equipment control')
@st.cache_resource
def api_client():
    client=requests.Session()
    if urlparse(API).hostname in ['127.0.0.1','localhost','::1']:client.trust_env=False
    return client
client=api_client()
def post(path,body):
    r=client.post(API+path,json=body,timeout=60);r.raise_for_status();return r.json()
try:ready=client.get(API+'/readyz',timeout=15).json()
except requests.RequestException:
    st.error('Inference service unavailable. Start with python deployment/serve.py or docker compose up -d.');st.stop()
with st.sidebar:
    st.header('Deployed model');st.write(ready.get('bundle') or 'Unavailable')
    st.caption('V0: source trained → V1: confident-label adaptation → V2: technician-label adaptation. Each input uses the active stage automatically.')
    st.warning('Three shared-weather source conditions. Target recordings are retrospective. All unverified settings require technician review.')
    cohort=st.selectbox('Feedback cohort',['test','genuine'],help='Use test for demonstrations. Genuine means actual operator feedback, not a reference-label simulation.')
if not ready.get('model_ready'):st.error('Model not ready: '+str(ready.get('error')))
view,comparison,learning,evidence=st.tabs(['Diagnosis','Experiments','Adaptation & feedback','Evidence'])
with view:
    source=st.radio('Input',['Recorded window','Upload canonical CSV'],horizontal=True)
    body={'cohort':cohort};route='/recorded';valid=True
    if source=='Recorded window':
        body['role']=st.selectbox('Recording pool',['target_adapt','target_feedback','source_validation','target_test'])
        body['index']=int(st.number_input('Window index',min_value=0,value=0,step=1))
    else:
        route='/infer';protocol=json.loads((ROOT/'data/aeris_windows_v1/window_protocol.json').read_text());features=protocol['core_features']
        st.caption('Exactly 15 consecutive one-minute rows. Temperatures in °C; commands 0–1. Uploads always require review because site applicability has not been established.')
        st.download_button('Download CSV template',pd.DataFrame(columns=['time_s']+features).to_csv(index=False),'aeris_input_template.csv')
        body['setting']=st.selectbox('AHU setting',['CoolingSeason','HeatingSeason','ShoulderSeason','SZVAV','SZCAV','Unknown'])
        file=st.file_uploader('Canonical sensor CSV',type='csv');valid=False
        if file:
            try:
                frame=pd.read_csv(file)
                if list(frame.columns)!=['time_s']+features or len(frame)!=15:raise ValueError('Require exact template columns and 15 rows; duplicate or extra columns are rejected.')
                body['raw']=frame[features].apply(pd.to_numeric,errors='raise').to_numpy().tolist();body['timestamps']=pd.to_numeric(frame.time_s,errors='raise').tolist()
                if not np.isfinite(np.asarray(body['raw'])).all():raise ValueError('Missing or nonfinite sensor data')
                valid=True
            except (ValueError,KeyError) as exc:st.error(str(exc))
    if st.button('Diagnose',disabled=not valid or not ready.get('model_ready'),type='primary'):
        started=time.perf_counter()
        try:
            record=post(route,{'request_id':uuid.uuid4().hex,**body});st.session_state['record']=record;st.session_state['ui_seconds']=time.perf_counter()-started
        except requests.RequestException as exc:st.error('Diagnosis request failed: '+str(exc))
    record=st.session_state.get('record')
    if record:
        r=record['result'];st.caption('Prediction '+record['prediction_id']+' · '+record['bundle']+' · '+record['cohort']+' · '+record['created'])
        if r['status'] in ['blocked','error']:st.error(' '.join(r['reasons']))
        else:
            a,b,c=st.columns(3);a.metric('Suggested condition',r['class_name'].replace('_',' ').title());b.metric('Model confidence',f"{r['confidence']:.1%}");c.metric('UI request round trip',f"{st.session_state['ui_seconds']*1000:.1f} ms")
            if r['reasons']:st.warning('Review required: '+' '.join(r['reasons']))
            elif r.get('fault_review_priority'):st.warning('Escalate for technician review: both branches support a known fault in this recorded research setting. Confirm the physical condition before action.')
            else:st.info('No confident known fault identified in this recorded research window. This is not clearance of the equipment; a technician must verify the physical condition before action.')
            st.bar_chart(pd.DataFrame({'Hybrid':r['probabilities'],'Forest':r['forest_probabilities'],'Temporal':r['temporal_probabilities']},index=CLASSES))
            with st.expander('Decision and prediction audit'):st.json(record)
            st.download_button('Export diagnosis audit',json.dumps(record,indent=2),'aeris_diagnosis.json')
            if st.button('Explain hybrid sensitivity'):
                try:st.session_state['explanation']=(record['prediction_id'],post('/explain',{'prediction_id':record['prediction_id']}))
                except requests.RequestException as exc:st.error(str(exc))
            exp=st.session_state.get('explanation')
            if exp and exp[0]==record['prediction_id']:
                st.dataframe(exp[1]['values']);st.caption(exp[1]['limitation'])
            with st.form('operator_feedback_'+record['prediction_id']):
                st.subheader('Technician review queue')
                agreement=st.selectbox('Agreement',['No response','agree','disagree','unsure'])
                satisfaction=st.selectbox('Satisfaction',['No response',1,2,3,4,5])
                corrected=st.selectbox('Override class',['No override']+CLASSES)
                decision=st.selectbox('Override decision',['No override','review_required','escalate_for_review','research_prediction'])
                note=st.text_area('Observation / inspection note')
                label=st.selectbox('Independently inspected label',['Unverified']+CLASSES)
                inspector=st.text_input('Inspector identity');provenance=st.text_input('Inspection evidence / work order reference')
                verified=st.checkbox('I independently verified this label by inspection, not by confidence or a published reference.')
                if st.form_submit_button('Save review record'):
                    if verified and (label=='Unverified' or not inspector.strip() or not provenance.strip()):st.error('Verified inspection requires a class, inspector and evidence provenance.')
                    else:
                        payload=dict(agreement=None if agreement=='No response' else agreement,satisfaction=None if satisfaction=='No response' else satisfaction,corrected_class=None if corrected=='No override' else corrected,corrected_decision=None if decision=='No override' else decision,note=note,verified_class=label if verified else None,inspector=inspector,provenance=provenance)
                        try:post('/feedback',{'prediction_id':record['prediction_id'],'feedback':payload});st.success('Review persisted. Eligible confirmed labels enter background V2 training; deployment waits for the evaluation gate.')
                        except requests.RequestException as exc:st.error('Feedback was not saved: '+str(exc))
with comparison:
    st.subheader('Measured experiments');current=ROOT/'experiments/end_to_end_20261116/before_after.csv'
    if current.exists():st.dataframe(pd.read_csv(current),hide_index=True)
    else:st.info('New experiments are in progress; historical results below are separate evidence.')
    historical=ROOT/'milestone4/evidence/results.csv'
    if historical.exists():st.caption('Historical M4 reports — these are not live prediction measurements.');st.dataframe(pd.read_csv(historical),hide_index=True)
with learning:
    st.subheader('Persistent feedback and reviewed updates')
    try:
        for k,s in client.get(API+'/summary',timeout=10).json().items():st.markdown('**'+k.title()+' cohort**');st.json(s)
    except requests.RequestException as exc:st.error(str(exc))
    st.info('Agreement and satisfaction are opinions. Confirmed eligible labels with identity and evidence enter background training in batches. Test learning is isolated from genuine use. Candidates must pass retention checks against both the parent and V0.')
    if st.checkbox('Load historical adaptation reports'):
        for p in (ROOT/'milestone4/model_records').glob('m4_*'):
            for f in ['selection_summary.json','feedback_report.json']:
                if (p/f).exists():
                    with st.expander('Historical '+p.name+' / '+f):st.json(json.loads((p/f).read_text()))
with evidence:
    if st.checkbox('Load detailed evidence'):
        for path in ['experiments/end_to_end_20261116/selection.json','experiments/end_to_end_20261116/audit.json','milestone4/evidence/readiness/scope.json']:
            if (ROOT/path).exists():
                with st.expander(path):st.json(json.loads((ROOT/path).read_text()))
    st.link_button('Original project repository','https://github.com/mharsha100/aeris-ahu-fault-diagnosis')
