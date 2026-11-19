"""Explicit reviewed genuine-feedback candidate; never promotes automatically.
Manifest: reviewed_by, parent_bundle, output_bundle, records[{prediction_id,
window_id, independent_case_id, partition: train|validation}]. All records must
be genuine, independently inspected and aligned to target_feedback windows.
"""
from pathlib import Path
import sys,json,argparse,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from sklearn.ensemble import RandomForestClassifier
from aeris.operations import Ledger
from aeris.features import model_inputs
from aeris.modeling import load_hybrid,CLASS_NAMES,metrics
from aeris.adaptation import fine_tune_temporal,recalibrate,source_guardrail
from aeris.workflow import write_bundle
from aeris.training import write_json,load_source_partition
def validate_records(ledger,manifest,metadata,feedback_ids):
    if not manifest.get('reviewed_by','').strip():raise ValueError('Named independent review is required')
    rows=[];seen=set();seen_windows=set();case_parts={};day_parts={}
    for item in manifest.get('records',[]):
        pid=item['prediction_id'];wid=item['window_id'];case=item['independent_case_id'];part=item['partition']
        if pid in seen or wid in seen_windows or part not in ['train','validation'] or not case:raise ValueError('Duplicate record/window or invalid partition/case')
        seen.add(pid);seen_windows.add(wid)
        if case in case_parts and case_parts[case]!=part:raise ValueError('Independent case crosses training and validation')
        case_parts[case]=part
        if wid not in feedback_ids:raise ValueError('Only reserved feedback-role windows may enter this reviewed update')
        day=metadata.loc[wid,'condition_group']
        if day in day_parts and day_parts[day]!=part:raise ValueError('Dependent recording day crosses feedback partitions')
        day_parts[day]=part
        with ledger.db() as c:r=c.execute('SELECT p.*,f.body FROM predictions p JOIN feedback f ON p.id=f.prediction_id WHERE p.id=?',(pid,)).fetchone()
        if not r or r['cohort']!='genuine':raise ValueError('Test, simulated or missing feedback is ineligible')
        f=json.loads(r['body'])
        if not f.get('verified_class') or not f.get('inspector') or not f.get('provenance'):raise ValueError('Independent inspection evidence is required')
        raw=np.asarray(json.loads(r['raw']),dtype='float32')
        if raw.shape!=(15,10) or not np.isfinite(raw).all():raise ValueError('Missing aligned sensor window')
        rows.append(dict(prediction_id=pid,window_id=wid,case=case,partition=part,label=CLASS_NAMES.index(f['verified_class']),raw=raw))
    if not rows:raise ValueError('No independently verified aligned genuine feedback exists')
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--db',required=True);args=p.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'));w=ROOT/'data/aeris_windows_v1';meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id');fp=dict(np.load(w/'target_feedback.npz'))
    rows=validate_records(Ledger(args.db),manifest,meta,set(fp['window_id']))
    parent_name=manifest['parent_bundle'];out_name=manifest['output_bundle']
    for n in [parent_name,out_name]:
        if not isinstance(n,str) or not n.replace('_','').replace('-','').isalnum():raise ValueError('Bundle names must be plain local identifiers')
    parent=ROOT/'data/aeris_models'/parent_name;model=load_hybrid(parent);torch.set_num_threads(2)
    for r in rows:
        i=list(fp['window_id']).index(r['window_id']);expected=(fp['X_seq'][i]*np.asarray(model.scaler['scale'])+model.scaler['mean']).astype('float32')
        np.testing.assert_allclose(r['raw'],expected,atol=3e-5)
    # Independent-case floor applies to both partitions and every class.
    for part in ['train','validation']:
        for k in range(4):
            if len({r['case'] for r in rows if r['partition']==part and r['label']==k})<5:raise ValueError('At least five independent cases per class in each feedback partition are required; windows do not substitute for cases')
    packs={}
    for part in ['train','validation']:
        rr=[r for r in rows if r['partition']==part];seq,tab=model_inputs(np.stack([r['raw'] for r in rr]),model.scaler,model.protocol);packs[part]=dict(X_seq=seq,X_tab=tab,y=np.array([r['label'] for r in rr]),confidence=np.ones(len(rr)))
    train,val,cal=[load_source_partition(w,r,model.protocol) for r in ['source_train','source_validation','source_calibration']]
    before=metrics(val['y'],model.predict(val['X_seq'],val['X_tab'])['probabilities']);fb_before=metrics(packs['validation']['y'],model.predict(packs['validation']['X_seq'],packs['validation']['X_tab'])['probabilities'])
    candidate=copy.deepcopy(model);candidate.forest=RandomForestClassifier(**model.forest.get_params());candidate.forest.n_jobs=2
    candidate.forest.fit(np.concatenate([train['X_tab'],packs['train']['X_tab']]),np.concatenate([train['y'],packs['train']['y']]))
    history,epoch=fine_tune_temporal(candidate.temporal,train,val,packs['train'],device='cpu',epochs=5,batch_size=256,seed=44);recalibrate(candidate,cal)
    after=metrics(val['y'],candidate.predict(val['X_seq'],val['X_tab'])['probabilities']);fb_after=metrics(packs['validation']['y'],candidate.predict(packs['validation']['X_seq'],packs['validation']['X_tab'])['probabilities']);guard=source_guardrail(before,after)
    def misses(m):return sum(m['confusion_matrix'][k][0] for k in [1,2,3])
    passed=guard['passed'] and misses(after)<=misses(before) and misses(fb_after)<=misses(fb_before) and fb_after['macro_f1']>=fb_before['macro_f1']
    output=ROOT/'data/aeris_models'/out_name
    write_bundle(output,candidate,parent,{'learning_stage':'verified_feedback','parent_bundle':parent_name,'feedback_status':'candidate_passed_review_required' if passed else 'rejected_do_not_promote','automatic_promotion':False})
    write_json(output/'feedback_report.json',{'passed':passed,'source_before':before,'source_after':after,'feedback_validation_before':fb_before,'feedback_validation_after':fb_after,'source_guard':guard,'selected_epoch':epoch,'reviewed_by':manifest['reviewed_by'],'parent_retained':True,'promoted':False});pd.DataFrame(history).to_csv(output/'feedback_history.csv',index=False)
    print('Candidate saved; parent deployment retained. Inspect feedback_report.json before any explicit promotion.')
if __name__=='__main__':main()
