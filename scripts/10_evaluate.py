"""Source-only selection followed by descriptive robustness, subgroup and latency audits."""
from pathlib import Path
import sys,json,time,copy,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from aeris.modeling import load_hybrid,metrics,CLASS_NAMES,sha256_file
from aeris.workflow import diagnose,canonical_window,SETTINGS
from aeris.operations import grouped_explanation
from aeris.training import write_json
from aeris.adaptation import adaptation_inputs,select_pseudo_labels,enough_candidates,POLICY
OUT=ROOT/'experiments/end_to_end_20261116';W=ROOT/'data/aeris_windows_v1'
def read(p):return json.loads(p.read_text())
def pack(role):return dict(np.load(W/f'{role}.npz',allow_pickle=False))
def policy(pred,y,seq,meta,threshold=.9):
    unsupported=np.array([int(k) not in SETTINGS.get(s,set()) for k,s in zip(pred['class_id'],meta.season)])
    review=(pred['confidence']<threshold)|pred['branch_disagreement']|(np.abs(seq).max(axis=(1,2))>6)|unsupported
    fault=y!=0;miss=fault&(pred['class_id']==0)
    return dict(review=review,accepted_false_normal=int((miss&~review).sum()),accepted_fault_denominator=int((fault&~review).sum()),review_windows=int(review.sum()),windows=len(y),coverage=float((~review).mean()),fault_to_normal=int(miss.sum()),fault_denominator=int(fault.sum()),diagnostic_fault_recall=1-float(miss.sum()/fault.sum()) if fault.any() else None,review_or_fault_escalation_recall=1-float((miss&~review).sum()/fault.sum()) if fault.any() else None)
def main():
    torch.set_num_threads(2)
    val=pack('source_validation');meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id').loc[val['window_id']].reset_index()
    results=[read(p) for p in OUT.glob('e2e_*/result.json')];rows=[];chosen={};decisions=[];cpu_predictions={};precision=[]
    # Select using the actual CPU deployment path. Preserve original CUDA trial results.
    for r in results:
        if not (r['name'].startswith('e2e_anchor_') or r['name'].startswith('e2e_final_')):continue
        cm=load_hybrid(ROOT/'data/aeris_models'/r['name']);cm.forest.n_jobs=2;cp=cm.predict(val['X_seq'],val['X_tab']);cpu_predictions[r['name']]=cp
        gp=dict(np.load(OUT/r['name']/'predictions.npz'));ms=metrics(val['y'],cp['probabilities'])
        precision.append(dict(bundle=r['name'],max_probability_difference=float(np.max(np.abs(cp['probabilities']-gp['probabilities']))),class_disagreements=int((cp['class_id']!=gp['class_id']).sum()),model_review_disagreements=int((cp['needs_review']!=gp['needs_review']).sum())))
        r['hybrid']={**r['hybrid'],'macro_f1':ms['macro_f1'],'log_loss':ms['log_loss'],'brier':ms['multiclass_brier'],'fault_to_normal':int(((val['y']!=0)&(cp['class_id']==0)).sum()),'normal_false_alarm':int(((val['y']==0)&(cp['class_id']!=0)).sum()),'recall':[ms['per_class'][c]['recall'] for c in CLASS_NAMES]}
    pd.DataFrame(precision).to_csv(OUT/'deployment_precision_audit.csv',index=False)
    for v in ['v0','v1','v2']:
        anchor=next(r for r in results if r['name']==f'e2e_anchor_{v}_s42')
        finalists=[r for r in results if r['name'].startswith(f'e2e_final_{v}')]
        if len(finalists)!=3:raise ValueError('Complete all three finalist seeds before selection')
        ap=cpu_predictions[anchor['name']];ag=policy(ap,val['y'],val['X_seq'],meta)
        eligible=True;reasons=[]
        for f in finalists:
            fp=cpu_predictions[f['name']];fg=policy(fp,val['y'],val['X_seq'],meta)
            ok=f['hybrid']['fault_to_normal']<=anchor['hybrid']['fault_to_normal'] and fg['accepted_false_normal']<=ag['accepted_false_normal'] and min(np.array(f['hybrid']['recall'])-np.array(anchor['hybrid']['recall']))>=-.03
            if not ok:eligible=False;reasons.append(f['name']+' failed false-normal/recall guard')
        mean=float(np.mean([r['hybrid']['macro_f1'] for r in finalists]));std=float(np.std([r['hybrid']['macro_f1'] for r in finalists]));worst=min(r['hybrid']['macro_f1'] for r in finalists)
        if mean<anchor['hybrid']['macro_f1']:eligible=False;reasons.append('Mean macro F1 did not exceed matched anchor')
        # Use fixed seed42 representative of a passing family, not a cherry-picked seed.
        winner=next(r for r in finalists if r['seed']==42) if eligible else anchor
        chosen[v]=winner['name'];decisions.append(dict(version=v,selected=winner['name'],finalist_mean=mean,finalist_std=std,finalist_worst=worst,retained=eligible,reasons=reasons,anchor=anchor['hybrid']))
        for r in [anchor,*finalists]:rows.append(dict(name=r['name'],version=v,seed=r['seed'],seconds=r['seconds'],**{k:z for k,z in r['hybrid'].items() if k!='recall'}))
    # The original verified October hybrid is also a fallback candidate.
    selection={'per_architecture':chosen,'decisions':decisions,'scope':'Source development selection only; all target recordings retrospective; independent case floor not met.'}
    # Rank a passing family by its three-seed mean, as declared before the runs.
    # A rejected family contributes only its verified anchor, not its finalist mean.
    def deployment_rank(d):
        r=next(r for r in results if r['name']==d['selected'])
        return (d['finalist_mean'] if d['retained'] else d['anchor']['macro_f1'],-r['hybrid']['fault_to_normal'])
    selected=max(decisions,key=deployment_rank)['selected']
    selection['bundle']=selected;selection['criteria']='selection_plan.json; all three seeds must pass safety guards. Fixed seed42 representative.'
    write_json(OUT/'selection.json',selection);pd.DataFrame(rows).to_csv(OUT/'before_after.csv',index=False)
    # Publish only after verified reload, and preserve every parent bundle.
    model=load_hybrid(ROOT/'data/aeris_models'/selected);model.forest.n_jobs=1
    write_json(ROOT/'data/deployment_selection.json',{'bundle':selected,'selection_evidence':'experiments/end_to_end_20261116/selection.json','previous_bundle':'v1_attention'})
    provenance={'source_packs':{r:sha256_file(W/f'{r}.npz') for r in ['source_train','source_validation','source_calibration']},'split_manifest':sha256_file(W/'split_manifest.csv'),'protocol':sha256_file(W/'window_protocol.json')};write_json(OUT/'provenance.json',provenance)
    subgroups=[];calibration=[];coverage=[];comparison=[];postselection=[];efficiency=[]
    for name in chosen.values():
        m=load_hybrid(ROOT/'data/aeris_models'/name);m.forest.n_jobs=2;p=m.predict(val['X_seq'],val['X_tab']);g=policy(p,val['y'],val['X_seq'],meta)
        for branch,key in [('hybrid','probabilities'),('RF-only','forest_probabilities'),('TNN-only','temporal_probabilities')]:
            mm=metrics(val['y'],p[key]);write_json(OUT/f'{name}_{branch}_metrics.json',mm);comparison.append({'model':name,'branch':branch,'macro_f1':mm['macro_f1'],'brier':mm['multiclass_brier'],'log_loss':mm['log_loss']})
        for group in ['season','dataset','condition_group','run_id']:
            for label,indices in meta.groupby(group).groups.items():
                ix=np.asarray(list(indices));y=val['y'][ix];pred=p['class_id'][ix]
                for k in range(4):
                    support=y==k
                    subgroups.append(dict(bundle=name,group_type=group,group=label,class_name=CLASS_NAMES[k],windows=int(support.sum()),recordings=int(meta.iloc[ix[support]].run_id.nunique()),independent_conditions_upper_bound=1 if group in ['season','condition_group','run_id'] else 3,evidence='insufficient independent cases (<5)',recall=float((pred[support]==k).mean()) if support.any() else None,fault_to_normal=int(((y!=0)&(pred==0)).sum()),fault_denominator=int((y!=0).sum()),review_rate=float(g['review'][ix].mean())))
        for t in [.5,.6,.7,.8,.85,.9,.95,.99]:
            q=policy(p,val['y'],val['X_seq'],meta,t);q.pop('review');coverage.append(dict(bundle=name,threshold=t,**q))
        for b in range(10):
            ix=(p['confidence']>=b/10)&(p['confidence']<=(b+1)/10 if b==9 else p['confidence']<(b+1)/10)
            calibration.append(dict(bundle=name,bin=b,support=int(ix.sum()),confidence=float(p['confidence'][ix].mean()) if ix.any() else None,accuracy=float((p['class_id'][ix]==val['y'][ix]).mean()) if ix.any() else None))
        # Scores below are opened only after source selection has been written.
        for role in ['source_test','target_test']:
            ep=pack(role);em=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id').loc[ep['window_id']].reset_index()
            ey=ep['y'] if role=='source_test' else pd.read_csv(W/'evaluation/target_window_reference.csv.gz').set_index('window_id').loc[ep['window_id'],'class_id'].to_numpy('int64')
            pp=m.predict(ep['X_seq'],ep['X_tab']);mm=metrics(ey,pp['probabilities']);write_json(OUT/f'{name}_{role}_descriptive.json',mm)
            postselection.append(dict(bundle=name,role=role,windows=len(ey),macro_f1=mm['macro_f1'],fault_to_normal=int(((ey!=0)&(pp['class_id']==0)).sum()),mandatory_review_fraction=1.0 if role=='target_test' else None,scope='retrospective target; never used for selection' if role=='target_test' else 'later source day within same conditions; not independent generalization'))
            if role=='target_test':
                for label,indices2 in em.groupby('dataset').groups.items():
                    ix2=np.asarray(list(indices2));y2=ey[ix2];pred2=pp['class_id'][ix2]
                    for k2 in range(4):
                        support2=y2==k2
                        subgroups.append(dict(bundle=name,group_type='target_setting',group=label,class_name=CLASS_NAMES[k2],windows=int(support2.sum()),recordings=int(em.iloc[ix2[support2]].run_id.nunique()),independent_conditions_upper_bound=int(em.iloc[ix2[support2]].condition_group.nunique()),evidence='retrospective; independence not established',recall=float((pred2[support2]==k2).mean()) if support2.any() else None,fault_to_normal=int(((y2!=0)&(pred2==0)).sum()),fault_denominator=int((y2!=0).sum()),review_rate=1.0))
        m.predict(val['X_seq'][:1],val['X_tab'][:1]);lat=[]
        for i in range(100):
            t=time.perf_counter();m.predict(val['X_seq'][i:i+1],val['X_tab'][i:i+1]);lat.append(time.perf_counter()-t)
        efficiency.append(dict(bundle=name,cpu_p50_seconds=float(np.quantile(lat,.5)),cpu_p95_seconds=float(np.quantile(lat,.95)),samples=100,scope='warm hybrid inference only; same CPU process; excludes UI',bytes=sum(f.stat().st_size for f in (ROOT/'data/aeris_models'/name).iterdir() if f.is_file())))
        target,tm=adaptation_inputs(W,m.protocol);tp=m.predict(target['X_seq'],target['X_tab']);a=select_pseudo_labels(target,tm,tp,m.protocol);enough,s= enough_candidates(a)
        # Applicability remains unverified for both real settings, regardless of high confidence.
        a['setting_support_pass']=False;a['final_rejection']='unsupported_target_setting';a.to_csv(OUT/f'{name}_adaptation_audit.csv.gz',index=False)
        write_json(OUT/f'{name}_adaptation.json',{'candidate_gate':s,'gate_rejections':a.selection_status.value_counts().to_dict(),'policy':POLICY,'status':'skipped_unsupported_setting','parent_retained':True,'updated_weights':False,'verified_feedback_stage':'skipped_no_independently_verified_aligned_labels','historical_reference_feedback':'Simulated; not independent evidence'})
    sg=pd.DataFrame(subgroups);sg['fault_to_normal_rate']=sg.fault_to_normal/sg.fault_denominator.replace(0,np.nan)
    pd.DataFrame(comparison).to_csv(OUT/'branch_comparison.csv',index=False);sg.to_csv(OUT/'subgroups.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False);pd.DataFrame(calibration).to_csv(OUT/'reliability.csv',index=False)
    pd.DataFrame(postselection).to_csv(OUT/'postselection_evaluation.csv',index=False);pd.DataFrame(efficiency).to_csv(OUT/'efficiency_comparison.csv',index=False)
    sp=model.predict(val['X_seq'],val['X_tab']);errors=meta.copy();errors['truth']=val['y'];errors['prediction']=sp['class_id'];errors['confidence']=sp['confidence'];errors['branch_disagreement']=sp['branch_disagreement'];errors['false_normal']=(val['y']!=0)&(sp['class_id']==0)
    errors=errors.loc[errors.truth!=errors.prediction].sort_values(['false_normal','confidence'],ascending=[False,False]);errors.to_csv(OUT/'selected_errors.csv.gz',index=False);errors.head(20).to_csv(OUT/'error_review_20.csv',index=False)
    training=pack('source_train');write_json(OUT/'selected_training_metrics.json',metrics(training['y'],model.predict(training['X_seq'],training['X_tab'])['probabilities']))
    rng=np.random.default_rng(20261116);indices=np.array([rng.choice(np.flatnonzero(val['y']==k),25,replace=False) for k in range(4)]).T.reshape(-1)
    rng=np.random.default_rng(20261116)  # Matched perturbations with scripts/17_baseline_controls.py.
    robustness=[];explanations=[];timings=[]
    for ix in indices:
        raw=canonical_window(val,int(ix),model.scaler);setting=meta.iloc[ix].season
        cases={'clean':raw.copy(),'noise_0.2C':raw.copy(),'offset_supply_1C':raw.copy(),'dropout':raw.copy(),'stuck_supply':raw.copy(),'unknown_setting':raw.copy()}
        cases['noise_0.2C'][:,:4]+=np.clip(rng.normal(0,.2,(15,4)),-.6,.6);cases['offset_supply_1C'][:,0]+=1;cases['dropout'][7,0]=np.nan;cases['stuck_supply'][:,0]=raw[0,0]
        for kind,changed in cases.items():
            start=time.perf_counter();r=diagnose(model,changed,setting='Unknown' if kind=='unknown_setting' else setting,timestamps=np.arange(15)*60,recorded_source=kind!='unknown_setting');seconds=time.perf_counter()-start
            robustness.append(dict(window_id=val['window_id'][ix],true_class=int(val['y'][ix]),case=kind,predicted=r.get('class_id'),status=r['status'],false_normal=bool(val['y'][ix]!=0 and r.get('class_id')==0),accepted_false_normal=bool(val['y'][ix]!=0 and r.get('class_id')==0 and r['status']=='research_prediction'),seconds=seconds))
            if kind=='clean':timings.append(seconds)
        if len(explanations)<6:
            start=time.perf_counter();e=grouped_explanation(model,raw);e2=grouped_explanation(model,raw);changed=raw.copy();changed[:,:4]+=.05;e3=grouped_explanation(model,changed)
            explanations.append(dict(window_id=val['window_id'][ix],truth=int(val['y'][ix]),seconds=(time.perf_counter()-start)/3,repeat_identical=e==e2,max_effect_change_under_005C=float(max(abs(a['probability_change_from_reference']-b['probability_change_from_reference']) for a,b in zip(e['values'],e3['values']))),explanation=e))
    rdf=pd.DataFrame(robustness);rdf.to_csv(OUT/'robustness_cases.csv',index=False);rdf.groupby('case').agg(windows=('window_id','size'),false_normal=('false_normal','sum'),accepted_false_normal=('accepted_false_normal','sum')).to_csv(OUT/'robustness_summary.csv')
    write_json(OUT/'explanations.json',explanations)
    model.predict(val['X_seq'][:32],val['X_tab'][:32]);start=time.perf_counter();model.predict(val['X_seq'][:1000],val['X_tab'][:1000]);batch=time.perf_counter()-start
    write_json(OUT/'efficiency.json',{'bundle':selected,'hardware': 'Windows CPU warm inference; CUDA training separately logged','warm_diagnose_100_p50_seconds':float(np.quantile(timings,.5)),'warm_diagnose_100_p95_seconds':float(np.quantile(timings,.95)),'batch_1000_seconds':batch,'throughput_windows_per_second':1000/batch,'bundle_bytes':sum(p.stat().st_size for p in (ROOT/'data/aeris_models'/selected).iterdir() if p.is_file()),'ui_latency_included':False})
    print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
