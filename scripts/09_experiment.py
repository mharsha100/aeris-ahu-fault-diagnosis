"""Resumable source-development experiment. Never uses target scores for selection."""
from pathlib import Path
import argparse, sys, json, time, random, copy, platform, traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch, joblib
from sklearn.ensemble import RandomForestClassifier
from aeris.modeling import *
from aeris.training import train_temporal, write_json, environment_check
from aeris.workflow import write_bundle

OUT=ROOT/'experiments/end_to_end_20261116'; OUT.mkdir(parents=True,exist_ok=True)
W=ROOT/'data/aeris_windows_v1'
K={'v0':'causal_tcn_last','v1':'causal_tcn_attention','v2':'causal_tcn_combined'}
def read(p): return json.loads(p.read_text())
def pack(role): return dict(np.load(W/f'{role}.npz',allow_pickle=False))
def score(y,p):
    m=metrics(y,p); pred=p.argmax(1); fault=y!=0
    return dict(macro_f1=m['macro_f1'],log_loss=m['log_loss'],brier=m['multiclass_brier'],
        fault_to_normal=int(((pred==0)&fault).sum()),fault_windows=int(fault.sum()),
        normal_false_alarm=int(((pred!=0)&~fault).sum()),
        recall=[m['per_class'][n]['recall'] for n in CLASS_NAMES],windows=len(y))
def audit():
    report={'environment':environment_check('cuda'),'m4_bundles_missing':[],'historical':{},'reproduction':{},'packs':{},'raw':[]}
    for v in K:
        if not (ROOT/f'data/aeris_models/m4_{v}/forest.joblib').exists(): report['m4_bundles_missing'].append('m4_'+v)
        report['historical'][v]=read(ROOT/f'milestone4/model_records/m4_{v}/training_run.json')
    for role in ['source_train','source_validation','source_calibration','source_test','target_adapt','target_feedback','target_test']:
        p=pack(role); report['packs'][role]={'sha256':sha256_file(W/f'{role}.npz'),'shape':list(p['X_seq'].shape),'finite':bool(np.isfinite(p['X_seq']).all()),'unique_ids':len(set(p['window_id'])),'classes':np.bincount(p['y'],minlength=4).tolist() if 'y' in p else None}
    meta=pd.read_csv(W/'window_metadata.csv.gz'); checks=[]
    for key,g in meta.groupby(['run_id','condition_group']):
        parts=g.groupby('partition').agg(start=('start_time_s','min'),end=('end_time_s','max'))
        rows=list(parts.itertuples())
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                checks.append({'group':str(key),'left':a.Index,'right':b.Index,'disjoint':bool(a.end<b.start or b.end<a.start)})
    report['interval_separation']=checks
    manifest=pd.read_csv(ROOT/'data/aeris_prepared/input_manifest.csv')
    contract=read(ROOT/'data/aeris_prepared/data_contract.json')
    for row in manifest.itertuples():
        path=ROOT/'data'/row.relative_path
        report['raw'].append({'path':row.relative_path,'present':path.exists(),'hash_matches':path.exists() and sha256_file(path)==row.sha256})
    for domain in ['source','target']:
        p=ROOT/f'data/aeris_prepared/{domain}_observations.csv.gz'
        df=pd.read_csv(p,low_memory=False)
        report[domain+'_observations']={'rows':len(df),'columns':list(df.columns),'missing':df[contract['core_features']].isna().sum().to_dict(),'duplicate_row_ids':int(df.row_id.duplicated().sum())}
    val=pack('source_validation')
    for name in ['v0','v1_attention']:
        m=load_hybrid(ROOT/'data/aeris_models'/name); m.forest.n_jobs=2
        pred=m.predict(val['X_seq'],val['X_tab']); hist=read(ROOT/'data/aeris_models'/name/'source_validation_metrics.json')
        report['reproduction'][name]={'current':score(val['y'],pred['probabilities']),'historical':hist,'bundle_sha256':sha256_file(ROOT/'data/aeris_models'/name/'hybrid.json')}
    report['dataset_mapping']={'source':'G36-1wk simulated multi-zone AHU','target':'SZVAV and SZCAV single-zone systems','AHU1_AHU2':'No authoritative numeric alias in contract; do not infer mapping.'}
    report['independent_case_limit']='Three shared-weather source conditions; days and overlapping windows are not independent fault injections. Five-case reporting floor not met.'
    write_json(OUT/'audit.json',report)
    meta.groupby(['domain','dataset','season','partition'],dropna=False).agg(windows=('window_id','size'),recordings=('run_id','nunique'),days=('condition_group','nunique')).to_csv(OUT/'subgroup_support.csv')
    print('AUDIT COMPLETE',flush=True)

BASE={'trees':200,'depth':18,'leaf':3,'features':'sqrt','rf_weight':'balanced_subsample','width':32,'dropout':.15,'lr':.001,'decay':.0001,'batch':256,'weighted':True,'dilations':[1,2,4]}
def trial(name,version,cfg,epochs,seed):
    folder=OUT/name; folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists(): return read(folder/'result.json')
    start=time.perf_counter(); torch.set_num_threads(2); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    write_json(folder/'config.json',dict(name=name,version=version,parameters=cfg,epochs=epochs,seed=seed,device='cuda',checkpoint='Best development macro F1; loss tie-break; no early stop',split='aeris-windows-v1 frozen source train/validation/calibration',code_sha256=sha256_file(Path(__file__))))
    train,val,cal=[pack(r) for r in ['source_train','source_validation','source_calibration']]
    protocol=read(W/'window_protocol.json'); scaler=read(W/'source_scaler.json')
    rf=RandomForestClassifier(n_estimators=cfg['trees'],max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features=cfg['features'],class_weight=cfg['rf_weight'],random_state=seed,n_jobs=4)
    rfstart=time.perf_counter(); rf.fit(train['X_tab'],train['y']); rfseconds=time.perf_counter()-rfstart
    arch=dict(input_features=10,channels=cfg['width'],dropout=cfg['dropout'],dilations=cfg['dilations'],classes=4)
    nn=create_temporal(K[version],arch).to('cuda')
    history,best=train_temporal(nn,train,val,device='cuda',epochs=epochs,batch_size=cfg['batch'],patience=0,seed=seed,learning_rate=cfg['lr'],weight_decay=cfg['decay'],class_weighted=cfg['weighted'],checkpoint_dir=folder)
    fv,fc=rf.predict_proba(val['X_tab']),rf.predict_proba(cal['X_tab'])
    tv,tc=[softmax(neural_logits(nn,p['X_seq'],'cuda')) for p in [val,cal]]
    ft,tt=fit_temperature(fc,cal['y']),fit_temperature(tc,cal['y'])
    fusion=[]
    for weight in [.25,.5,.75,.9]:
        p=weight*temperature_probabilities(fv,ft)+(1-weight)*temperature_probabilities(tv,tt)
        fusion.append({'weight':weight,**score(val['y'],p)})
    selected=max(fusion,key=lambda r:(r['macro_f1'],-r['fault_to_normal'],-r['log_loss']))
    weight=selected['weight']; ht=fit_temperature(weight*temperature_probabilities(fc,ft)+(1-weight)*temperature_probabilities(tc,tt),cal['y'])
    config=dict(artifact_version='aeris-hybrid-v1',model_version=version,architecture_id=K[version],temporal_kind=K[version],temporal_architecture=arch,classes=CLASS_NAMES,learning_stage='initial_source_training',initialization='fresh',parent_weights_loaded=False,forest_weight=weight,temporal_weight=1-weight,temperatures=dict(forest=ft,temporal=tt,hybrid=ht),review_confidence_threshold=.9,selected_temporal_epoch=best,seed=seed,target_data_used=False,test_data_used=False,run_name=name,training_parameters=cfg)
    model=HybridPredictor(rf,nn,config,scaler,protocol,'cuda'); pred=model.predict(val['X_seq'],val['X_tab'])
    uncal=weight*fv+(1-weight)*tv
    np.savez_compressed(folder/'predictions.npz',y=val['y'],window_id=val['window_id'],**pred)
    result=dict(name=name,version=version,seed=seed,epochs=epochs,best_epoch=best,parameters=cfg,seconds=time.perf_counter()-start,rf_seconds=rfseconds,fusion=fusion,selected_weight=weight,hybrid=score(val['y'],pred['probabilities']),rf=score(val['y'],fv),temporal=score(val['y'],tv),uncalibrated=score(val['y'],uncal),status='complete')
    write_json(folder/'pending_result.json',result)
    output=ROOT/'data/aeris_models'/name
    restored=write_bundle(output,model,W,{'run_evidence':str(folder.relative_to(ROOT))})
    # Compare serialization on the same CPU and batch shape. CUDA-vs-CPU is a separate numerical measurement.
    cpu_after=restored.predict(val['X_seq'][:32],val['X_tab'][:32])['probabilities']
    model.temporal.to('cpu');model.device='cpu'
    cpu_before=model.predict(val['X_seq'][:32],val['X_tab'][:32])['probabilities']
    np.testing.assert_allclose(cpu_after,cpu_before,atol=1e-7)
    result['reload_cpu_max_abs_difference']=float(np.max(np.abs(cpu_after-cpu_before)))
    result['cpu_cuda_max_abs_difference_32']=float(np.max(np.abs(cpu_after-pred['probabilities'][:32])))
    result['cpu_cuda_class_disagreements_32']=int((cpu_after.argmax(1)!=pred['probabilities'][:32].argmax(1)).sum())
    result['bundle_sha256']=sha256_file(output/'hybrid.json'); write_json(folder/'result.json',result)
    with (OUT/'trials.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
    print('TRIAL COMPLETE '+name+' '+str(result['hybrid']),flush=True)
    return result

def run(mode):
    if mode=='audit': return audit()
    if mode=='pilot': return trial('pilot_v0_s42','v0',BASE,2,42)
    pilot=read(OUT/'pilot_v0_s42/result.json')
    plan={'screen_trials':6,'screen_epochs':10,'anchor_trials':3,'anchor_epochs':50,'finalist_seeds':[42,43,44],'finalist_epochs':50,'max_seconds':14400,'pilot_seconds':pilot['seconds'],'criterion':'Source validation macro F1 primary. Relative to the matched anchor: no increase in fault-to-normal count or accepted false-normal count; no class recall drop >0.03. All thresholds provisional. Mandatory scope review. Choose within each architecture; deployment best eligible mean F1, then fewer false-normal errors. Retain anchor if no candidate passes.','target_scores_for_selection':False,'hypotheses':['False-normal errors motivate RF leaf/depth/class balance and unweighted loss comparison.','Attention may overfit shared recordings: test width/dropout/weight decay.','Existing features already include rolling mean/std/min/max/last/trend; no missingness imputation without validation.','15-sample windows with RF=0.5 need fusion sensitivity and calibration checks.']}
    write_json(OUT/'selection_plan.json',plan); started=time.monotonic(); results=[]
    candidates=[dict(BASE,trees=100,depth=12,leaf=1,features=.7,rf_weight=None,width=16,dropout=.05,lr=.0005,decay=.00001,batch=128,weighted=False),dict(BASE,trees=300,depth=24,leaf=6,features=.5,width=48,dropout=.3,lr=.002,decay=.001,batch=512)]
    for v in K:
        anchors=trial(f'e2e_anchor_{v}_s42',v,BASE,50,42); results.append(anchors)
        screens=[]
        for i,c in enumerate(candidates):
            screens.append(trial(f'e2e_screen_{v}_{i}_s42',v,c,10,42))
        chosen=max(screens,key=lambda r:r['hybrid']['macro_f1'])
        for seed in [42,43,44]:
            if time.monotonic()-started>plan['max_seconds']: raise TimeoutError('Documented experiment time budget reached; completed trials retained')
            results.append(trial(f'e2e_final_{v}_s{seed}',v,chosen['parameters'],50,seed))
    write_json(OUT/'completed.json',{'runs':[r['name'] for r in results],'elapsed_seconds':time.monotonic()-started})
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['audit','pilot','search']); args=p.parse_args()
    try: run(args.mode)
    except Exception as e:
        with (OUT/'failures.jsonl').open('a') as f: f.write(json.dumps({'mode':args.mode,'error':str(e),'traceback':traceback.format_exc()})+'\n')
        raise
