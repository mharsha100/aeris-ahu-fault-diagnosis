"""Versioned research controls and auditable verified-feedback updates."""
from pathlib import Path
import copy, json, os, random, time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from .modeling import CLASS_NAMES, load_hybrid, HybridPredictor, metrics, sha256_file
from .features import model_inputs
from .training import load_source_partition, write_json
from .adaptation import fine_tune_temporal, recalibrate, source_guardrail

POLICY_VERSION = 'm4-research-review-v2'
SETTINGS = {'CoolingSeason': {0,1,2}, 'HeatingSeason': {0,1,3}, 'ShoulderSeason': {0,1,2}}


def restore_packaged_forests(root):
    """Restore deduplicated exact artifact bytes in downloadable releases."""
    import shutil
    root=Path(root);manifest=root/'data/model_install.json'
    if not manifest.exists():return
    for bundle,sha in json.loads(manifest.read_text()).items():
        if bundle not in {f'm4_v{k}{s}' for k in range(3) for s in ['', '_cross_adapted','_feedback']}:
            raise ValueError('Unexpected bundle name in installation manifest')
        dest=root/'data/aeris_models'/bundle/'forest.joblib'
        if not dest.exists():
            source=root/'data/aeris_model_blobs'/f'{sha}.joblib'
            if not source.is_file() or sha256_file(source)!=sha:raise ValueError('Missing or corrupted packaged forest')
            shutil.copyfile(source,dest)


def diagnose(model, raw, *, setting, timestamps=None, recorded_source=False):
    """One canonical 15-minute window; invalid input never produces a diagnosis."""
    raw=np.asarray(raw,dtype='float32')
    if raw.shape != (15,10) or not np.isfinite(raw).all():
        return {'status':'blocked','reasons':['Require 15 complete rows of 10 finite sensor values.'], 'policy_version':POLICY_VERSION}
    if np.any(raw[:,:4]<-60) or np.any(raw[:,:4]>100) or np.any(raw[:,5:9]<-1e-6) or np.any(raw[:,5:9]>1+1e-6):
        return {'status':'blocked','reasons':['Sensor values exceed physical input bounds.'], 'policy_version':POLICY_VERSION}
    if np.any(raw[:,[4,9]]!=1):
        return {'status':'blocked','reasons':['This model requires occupied operation with the supply fan on.'], 'policy_version':POLICY_VERSION}
    if timestamps is not None:
        t=np.asarray(timestamps,dtype='float64')
        if t.shape!=(15,) or not np.isfinite(t).all() or not np.allclose(np.diff(t),60,rtol=0,atol=1e-6):
            return {'status':'blocked','reasons':['Timestamps must be 15 consecutive one-minute samples.'], 'policy_version':POLICY_VERSION}
    seq,tab=model_inputs(raw,model.scaler,model.protocol)
    prediction=model.predict(seq,tab)
    k=int(prediction['class_id'][0]); conf=float(prediction['confidence'][0]); reasons=[]
    if not recorded_source and setting in SETTINGS: reasons.append('Uploaded or unverified AHU context: recorded-source applicability is not established.')
    if setting not in SETTINGS: reasons.append('Exploratory or unknown AHU setting: transfer has not been validated.')
    elif k not in SETTINGS[setting]: reasons.append('Predicted class is outside observed class/season coverage.')
    if model.configuration.get('learning_stage')=='verified_feedback': reasons.append('Experimental reference-feedback candidate: source retention is not operational approval.')
    if conf<0.90: reasons.append('Confidence is below the provisional 0.90 review threshold.')
    if bool(prediction['branch_disagreement'][0]): reasons.append('Classical and temporal branches disagree.')
    out=int((np.abs(seq[0]).max(axis=1)>6).sum())
    if out: reasons.append(f'{out}/15 samples exceed the provisional source-scale support bound (6 standard deviations).')
    return {'status':'review_required' if reasons else 'research_prediction',
            'class_id':k,'class_name':CLASS_NAMES[k],'confidence':conf,
            'probabilities':prediction['probabilities'][0].tolist(),
            'forest_probabilities':prediction['forest_probabilities'][0].tolist(),
            'temporal_probabilities':prediction['temporal_probabilities'][0].tolist(),
            'reasons':reasons,'source_support_warning_samples':out,
            'policy_version':POLICY_VERSION,'automatic_actuation_allowed':False,
            'fault_review_priority': bool(k!=0 and not reasons),
            'limitation':'Research decision support. Confidence is not verified target accuracy.'}


def canonical_window(pack, index, scaler):
    return (pack['X_seq'][index].astype('float64')*np.asarray(scaler['scale'])+np.asarray(scaler['mean'])).astype('float32')


def write_bundle(output, predictor, parent, extra):
    output=Path(output)
    if output.exists(): raise FileExistsError(f'Preserve completed output: {output}')
    output.mkdir(parents=True)
    config=copy.deepcopy(predictor.configuration);config.update(extra)
    joblib.dump(predictor.forest,output/'forest.joblib',compress=3)
    torch.save({k:v.detach().cpu() for k,v in predictor.temporal.state_dict().items()},output/'temporal_state.pt')
    for name in ['source_scaler.json','window_protocol.json']:(output/name).write_bytes((parent/name).read_bytes())
    config['artifact_sha256']={name:sha256_file(output/name) for name in ['forest.joblib','temporal_state.pt','source_scaler.json','window_protocol.json']}
    write_json(output/'hybrid.json',config)
    write_json(output/'READY.json',{'model_version':config['model_version'],'hybrid_sha256':sha256_file(output/'hybrid.json')})
    return load_hybrid(output)


def choose_feedback(pack, meta, pred, limit=20):
    """Freeze IDs from inputs/predictions alone, before opening published labels."""
    audit=meta.copy();audit['pack_index']=np.arange(len(meta));audit['confidence']=pred['confidence'];audit['disagree']=pred['branch_disagreement']
    audit=audit.sort_values(['disagree','confidence','window_id'],ascending=[False,True,True])
    chosen=[];days={};intervals={}
    for row in audit.itertuples():
        if len(chosen)>=limit:break
        if days.get(row.condition_group,0)>=3:continue
        key=(row.run_id,row.condition_group)
        if any(row.start_time_s<=b and row.end_time_s>=a for a,b in intervals.get(key,[])):continue
        chosen.append(row.pack_index);days[row.condition_group]=days.get(row.condition_group,0)+1
        intervals.setdefault(key,[]).append((row.start_time_s,row.end_time_s))
    return np.asarray(chosen,dtype='int64')


def feedback_update(root, name, *, epochs=5, seed=44):
    """Simulated technician: published reference supplies only preselected labels."""
    root=Path(root);windows=root/'data/aeris_windows_v1';parent=root/'data/aeris_models'/f'{name}_cross_adapted';output=parent.with_name(f'{name}_feedback')
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    baseline=load_hybrid(parent);baseline.forest.n_jobs=2
    pack={k:v for k,v in np.load(windows/'target_feedback.npz',allow_pickle=False).items()}
    if 'y' in pack:raise ValueError('Feedback selection must not have labels')
    meta=pd.read_csv(windows/'window_metadata.csv.gz');meta=meta.set_index('window_id').loc[pack['window_id']].reset_index()
    if not meta.partition.eq('target_feedback').all():raise ValueError('Wrong feedback role')
    pred=baseline.predict(pack['X_seq'],pack['X_tab']);ix=choose_feedback(pack,meta,pred)
    selection_path=root/'milestone4/evidence'/f'{name}_feedback_selection.json'
    selection_path.parent.mkdir(exist_ok=True,parents=True)
    write_json(selection_path,{'selected_window_ids':pack['window_id'][ix].tolist(),'labels_opened_during_selection':False,
                              'selection':'Disagreement first, confidence ascending, max 3/day, nonoverlapping windows, budget20'})
    # Only after frozen selection, reveal selected public reference labels.
    ref=pd.read_csv(windows/'evaluation/target_window_reference.csv.gz').set_index('window_id')
    selected=ref.loc[pack['window_id'][ix]]
    label_col='class_id'
    y=selected[label_col].to_numpy(dtype='int64')
    if not set(y).issubset({0,1,2,3}):raise ValueError('Invalid verified class')
    pseudo={'X_seq':pack['X_seq'][ix].astype('float32'),'X_tab':pack['X_tab'][ix].astype('float32'),'y':y,'confidence':np.ones(len(ix))}
    train=load_source_partition(windows,'source_train',baseline.protocol);val=load_source_partition(windows,'source_validation',baseline.protocol);cal=load_source_partition(windows,'source_calibration',baseline.protocol)
    if set(pack['window_id'][ix]) & (set(train['window_id'])|set(val['window_id'])|set(cal['window_id'])):raise ValueError('Feedback/source overlap')
    before=metrics(val['y'],baseline.predict(val['X_seq'],val['X_tab'])['probabilities'])
    forest=RandomForestClassifier(**baseline.forest.get_params())
    forest.fit(np.concatenate([train['X_tab'],pseudo['X_tab']]),np.concatenate([train['y'],y]),sample_weight=np.r_[np.ones(len(train['y'])),np.full(len(y),5.)])
    temporal=copy.deepcopy(baseline.temporal)
    history,epoch=fine_tune_temporal(temporal,train,val,pseudo,device='cpu',epochs=epochs,batch_size=256,seed=seed)
    candidate=HybridPredictor(forest,temporal,copy.deepcopy(baseline.configuration),baseline.scaler,baseline.protocol,'cpu');recalibrate(candidate,cal)
    after=metrics(val['y'],candidate.predict(val['X_seq'],val['X_tab'])['probabilities']);guard=source_guardrail(before,after)
    accepted=guard['passed'];deployed=candidate if accepted else baseline
    status='accepted_source_guard' if accepted else 'rejected_source_guard_parent_retained'
    restored=write_bundle(output,deployed,parent,{'learning_stage':'verified_feedback','feedback_status':status,'parent_bundle':parent.name,
            'parent_hybrid_sha256':sha256_file(parent/'hybrid.json'),'feedback_labels':len(ix),
            'feedback_provenance':'Simulated technician from pre-existing published reference; not real technician intervention',
            'target_labels_used':True,'target_data_used':bool(accepted or baseline.configuration.get('target_data_used')),'test_data_used':False,'automatic_actuation_allowed':False})
    p1=deployed.predict(val['X_seq'][:32],val['X_tab'][:32])['probabilities'];p2=restored.predict(val['X_seq'][:32],val['X_tab'][:32])['probabilities'];np.testing.assert_allclose(p1,p2,atol=1e-6)
    selected=meta.iloc[ix].copy();selected['verified_class_id']=y;selected['verification_source']='published_reference_simulated_technician';selected.to_csv(output/'verified_feedback.csv',index=False)
    pd.DataFrame(history).to_csv(output/'feedback_training_history.csv',index=False)
    write_json(output/'feedback_report.json',{'status':status,'selected_windows':len(ix),'selected_days':int(selected.condition_group.nunique()),
        'selection_labels_opened':False,'simulated_technician':True,'unverified_notes_used_for_training':False,
        'completed_epochs':len(history),'retained_temporal_epoch':epoch,'seed':seed,'source_guardrail':guard,
        'source_before':before,'source_candidate':after,'saved_reload_max_abs_difference':float(np.max(np.abs(p1-p2))),
        'target_feedback_sha256':sha256_file(windows/'target_feedback.npz'),'selection_sha256':sha256_file(selection_path),
        'label_reference_sha256':sha256_file(windows/'evaluation/target_window_reference.csv.gz'),'final_test_labels_used':False,
        'note':'Source retention does not establish target benefit. Rejected updates preserve parent; all target evaluation is retrospective exploratory.'})
    write_json(output/'training_run.json',{'learning_stage':'verified_feedback','parent_bundle':parent.name,'parent_weights_loaded':True,
        'completed_feedback_epochs':len(history),'seed':seed,'feedback_status':status,'source_pack_sha256':{r:sha256_file(windows/f'{r}.npz') for r in ['source_train','source_validation','source_calibration']},
        'code_sha256':sha256_file(Path(__file__)),'target_test_used_for_fitting':False})
    print(f'{name} feedback: {len(ix)} verified reference labels; {status}',flush=True)
    return output
