"""Bounded, defensive black-box sensor attacks and full-hybrid explanation checks."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from aeris.modeling import load_hybrid,metrics
from aeris.operations import grouped_explanation
from aeris.workflow import canonical_window,diagnose,SETTINGS
from aeris.training import write_json
OUT=ROOT/'experiments/operator_redesign_20261117';W=ROOT/'data/aeris_windows_v1'
def main():
    torch.set_num_threads(2);name='boosted_v2_s42';m=load_hybrid(ROOT/'data/aeris_models'/name);m.forest.n_jobs=1
    pack=dict(np.load(W/'source_validation.npz'));meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id').loc[pack['window_id']].reset_index();rng=np.random.default_rng(20261117)
    ix=np.array([rng.choice(np.flatnonzero(pack['y']==k),10,replace=False) for k in range(4)]).T.reshape(-1);attacks=[];explanations=[]
    plan={'bundle':name,'seed':20261117,'windows':40,'class_support':[10,10,10,10],'candidates_per_window_and_budget':65,'budgets':[{'temperature_C':.2,'commands':.01},{'temperature_C':.5,'commands':.02},{'temperature_C':1.,'commands':.05}],'attack':'64 bounded randomized constant offsets plus time noise; clean input included as candidate 0. Untargeted minimum true-class probability and separate targeted normal search.','constraints':'Temperatures bounded to +/- budget around original and [-60,100] C; commands within +/- budget and [0,1]. Fan/occupancy/timestamps unchanged. Both branches derived from the same modified raw input.','scope':'Offline defensive source-development experiment. Not a certified worst-case guarantee, field adversarial validation, or target transfer claim.'};write_json(OUT/'attack_plan.json',plan)
    for idx in ix:
        raw=canonical_window(pack,int(idx),m.scaler);y=int(pack['y'][idx]);season=meta.iloc[idx].season;clean=m.predict_raw(raw);clean_correct=int(clean['class_id'][0])==y
        for eps,command_eps in [(.2,.01),(.5,.02),(1.,.05)]:
            windows=np.repeat(raw[None],65,axis=0)
            delta=np.clip(rng.uniform(-eps,eps,(64,1,4))+rng.normal(0,eps/5,(64,15,4)),-eps,eps)
            windows[1:,:,:4]=np.clip(windows[1:,:,:4]+delta,-60,100)
            windows[1:,:,5:9]=np.clip(windows[1:,:,5:9]+rng.uniform(-command_eps,command_eps,(64,1,4)),0,1)
            p=m.predict_raw(windows);z=(windows-np.asarray(m.scaler['mean']))/np.asarray(m.scaler['scale'])
            review=p['needs_review']|(np.abs(z)>6).any((1,2))|~np.isin(p['class_id'],list(SETTINGS[season]));chosen=int(p['probabilities'][:,y].argmin());target=int(p['probabilities'][:,0].argmax())
            wrong=(p['class_id']!=y);accepted_wrong=wrong&~review;normal_wrong=(p['class_id']==0)&(y!=0)
            attacks.append(dict(window_id=str(pack['window_id'][idx]),truth=y,temperature_budget=eps,command_budget=command_eps,clean_correct=clean_correct,worst_correct=not bool(wrong.any()),attack_success_on_clean_correct=bool(clean_correct and wrong.any()),any_accepted_wrong_class=bool(accepted_wrong.any()),any_accepted_false_normal=bool((normal_wrong&~review).any()),targeted_normal_class=int(p['class_id'][target]),worst_review=bool(review[chosen]),true_class_probability_clean=float(clean['probabilities'][0,y]),true_class_probability_worst=float(p['probabilities'][chosen,y])))
        if len(explanations)<20:
            t=time.perf_counter();e=grouped_explanation(m,raw);repeat=grouped_explanation(m,raw);modified=raw.copy();modified[:,:4]+=.05;e2=grouped_explanation(m,modified,class_id=int(clean['class_id'][0]));a=np.array([v['shapley_value'] for v in e['values']]);b=np.array([v['shapley_value'] for v in e2['values']]);rho=float(spearmanr(a,b).statistic)
            explanations.append(dict(window_id=str(pack['window_id'][idx]),truth=y,additivity_error=e['additivity_error'],repeat_identical=e==repeat,top_group_stable=int(abs(a).argmax())==int(abs(b).argmax()),spearman=rho if np.isfinite(rho) else None,maximum_contribution_change=float(abs(a-b).max()),seconds_per_explanation=(time.perf_counter()-t)/3))
        print('ASSURANCE',idx,flush=True)
    d=pd.DataFrame(attacks);d.to_csv(OUT/'adversarial_cases.csv',index=False);summary=[]
    for eps,g in d.groupby('temperature_budget'):
        correct=int(g.clean_correct.sum());summary.append(dict(temperature_budget_C=eps,command_budget=float(g.command_budget.iloc[0]),windows=len(g),fault_windows=int((g.truth!=0).sum()),clean_accuracy=float(g.clean_correct.mean()),attack_accuracy=float(g.worst_correct.mean()),clean_correct_windows=correct,successful_attacks=int(g.attack_success_on_clean_correct.sum()),attack_success_rate=float(g.attack_success_on_clean_correct.sum()/correct) if correct else None,windows_with_any_accepted_wrong_class=int(g.any_accepted_wrong_class.sum()),windows_with_any_accepted_false_normal=int(g.any_accepted_false_normal.sum()),worst_case_review_rate=float(g.worst_review.mean())))
    pd.DataFrame(summary).to_csv(OUT/'adversarial_summary.csv',index=False)
    write_json(OUT/'explanation_checks.json',{'bundle':name,'cases':explanations,'max_additivity_error':max(e['additivity_error'] for e in explanations),'repeatability':float(np.mean([e['repeat_identical'] for e in explanations])),'top_group_stability_under_005C':float(np.mean([e['top_group_stable'] for e in explanations])),'mean_seconds':float(np.mean([e['seconds_per_explanation'] for e in explanations])),'human_usefulness':'Not measured; requires independent technician assessment.','causality':'Not established by Shapley values.'})
    collisions=[];raw=pack['X_seq']*np.asarray(m.scaler['scale'])+m.scaler['mean'];y=pack['y']
    for eps in [0,.001,.01,.1]:
        x=raw if eps==0 else np.round(raw/np.array([eps]*4+[.001]*6));_,inv=np.unique(x.reshape(len(y),-1),axis=0,return_inverse=True);counts=np.zeros((inv.max()+1,4),int);np.add.at(counts,(inv,y),1);ambiguous=(counts>0).sum(1)>1
        collisions.append(dict(temperature_rounding_C=eps,command_rounding=0 if eps==0 else .001,windows=len(y),cross_label_collision_windows=int(counts[ambiguous].sum()),unique_windows=len(counts),best_possible_lookup_correct_on_these_quantized_groups=int(counts.max(1).sum()),class_collision_support=counts[ambiguous].sum(0).tolist()))
    write_json(OUT/'observability_audit.json',{'results':collisions,'interpretation':'This is an illustrative sensor-resolution study, not a bound on the original full-precision model or a proof of measurement uncertainty. Scenario labels can describe imposed faults that are inactive at a timestamp. Source validation only; not used to alter labels or choose a target model.'})
    print('ASSURANCE COMPLETE',flush=True)
if __name__=='__main__':main()

