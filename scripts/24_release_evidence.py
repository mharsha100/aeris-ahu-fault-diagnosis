"""Apply the preregistered source-only gates and publish measured release evidence."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from aeris.modeling import load_hybrid,metrics,CLASS_NAMES
from aeris.workflow import SETTINGS
from aeris.training import write_json
OUT=ROOT/'experiments/improvement_20261117';OLD=ROOT/'experiments/end_to_end_20261116';W=ROOT/'data/aeris_windows_v1'

def assess(name,pack,meta):
    m=load_hybrid(ROOT/'data/aeris_models'/name);m.forest.n_jobs=1;p=m.predict(pack['X_seq'],pack['X_tab']);y=pack['y'];c=p['class_id']
    review=(p['confidence']<.9)|p['branch_disagreement']|(np.abs(pack['X_seq'])>6).any((1,2))|np.array([int(k) not in SETTINGS[s] for k,s in zip(c,meta.season)])
    r=metrics(y,p['probabilities']);r.update(bundle=name,fault_to_normal=int(((y!=0)&(c==0)).sum()),accepted_false_normal=int(((y!=0)&(c==0)&~review).sum()),review_fraction=float(review.mean()),accepted_windows=int((~review).sum()),accepted_accuracy=float((c[~review]==y[~review]).mean()) if (~review).any() else None,accepted_normal_windows=int(((c==0)&~review).sum()),normal_windows_reviewed=int(((y==0)&review).sum()))
    write_json(OUT/(name+'_release_metrics.json'),r);return r

def main():
    torch.set_num_threads(2)
    assert (OUT/'hybrids_complete.json').exists(),'Finish all nine training runs first'
    assert len(list(OUT.glob('calibrated_physics_*/result.json')))==9,'Finish all nine calibration fits first'
    pack=dict(np.load(W/'source_validation.npz'));meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id').loc[pack['window_id']].reset_index()
    old=json.loads((OLD/'selection.json').read_text());anchors={v:assess(n,pack,meta) for v,n in old['per_architecture'].items()};trials=[];decisions=[];selected=dict(old['per_architecture'])
    for version in ['v0','v1','v2']:
        anchor=anchors[version];qualified=[]
        for family in ['physics','calibrated_physics']:
            group=[]
            for seed in [42,43,44]:
                name=f'{family}_{version}_s{seed}';r=assess(name,pack,meta);recall_drop=max(anchor['per_class'][c]['recall']-r['per_class'][c]['recall'] for c in CLASS_NAMES);reasons=[]
                if r['macro_f1']<anchor['macro_f1']+.01:reasons.append('macro F1 gain below 0.01')
                if r['fault_to_normal']>anchor['fault_to_normal']:reasons.append('more faults predicted normal')
                if r['accepted_false_normal']>anchor['accepted_false_normal']:reasons.append('more accepted false-normal decisions')
                if recall_drop>.03:reasons.append('class recall drops more than 0.03')
                row=dict(name=name,version=version,family=family,seed=seed,macro_f1=r['macro_f1'],accuracy=r['accuracy'],fault_to_normal=r['fault_to_normal'],accepted_false_normal=r['accepted_false_normal'],maximum_recall_drop=recall_drop,passed=not reasons,reasons='; '.join(reasons));group.append(row);trials.append(row)
            passed=all(r['passed'] for r in group);decision=dict(version=version,family=family,passed_all_seeds=passed,mean_f1=float(np.mean([r['macro_f1'] for r in group])),std_f1=float(np.std([r['macro_f1'] for r in group])),worst_f1=min(r['macro_f1'] for r in group),reasons=[r['name']+': '+r['reasons'] for r in group if not r['passed']]);decisions.append(decision)
            if passed:qualified.append(decision)
        if qualified:selected[version]=max(qualified,key=lambda x:x['mean_f1'])['family']+'_'+version+'_s42'
    pd.DataFrame(trials).to_csv(OUT/'release_trials.csv',index=False)
    # Freeze selection before reading any later-day or target evaluation.
    selection=dict(bundle=selected['v2'],per_architecture=selected,decisions=decisions,criteria='hybrid_plan.json and class_decision_plan.json; every seed must pass; fixed seed42 representative',scope='Source development only. Target labels were previously exposed; no independent field generalization claim.')
    write_json(OUT/'selection.json',selection)
    if selected!=old['per_architecture']:
        raise RuntimeError('New candidate passed: run separate post-selection source/target evaluation before deployment')
    write_json(ROOT/'data/release_catalog.json',{'models':[dict(version=v,bundle=n,stage='Retained after guarded comparison') for v,n in selected.items()]})
    write_json(ROOT/'data/deployment_selection.json',selection)
    comparison=[];classes=[]
    for v,n in selected.items():
        r=anchors[v]
        comparison.append(dict(version=v.upper(),partition='Source development',n=r['windows'],accuracy=r['accuracy'],macro_F1=r['macro_f1'],macro_precision=r['per_class']['macro avg']['precision'],macro_recall=r['per_class']['macro avg']['recall'],ECE=r['ece_10_bins'],Brier=r['multiclass_brier'],review_rate=r['review_fraction']))
        for c in CLASS_NAMES:classes.append(dict(version=v.upper(),condition=c,**{k:r['per_class'][c][k] for k in ['precision','recall','f1-score','support']}))
        for role,label in [('source_test','Later source day'),('target_test','Retrospective target')]:
            h=json.loads((OLD/f'{n}_{role}_descriptive.json').read_text());comparison.append(dict(version=v.upper(),partition=label,n=h['windows'],accuracy=h['accuracy'],macro_F1=h['macro_f1'],macro_precision=h['per_class']['macro avg']['precision'],macro_recall=h['per_class']['macro avg']['recall'],ECE=h['ece_10_bins'],Brier=h['multiclass_brier'],review_rate=1. if role=='target_test' else None))
    attacks=pd.read_csv(OUT/'adversarial_summary.csv');e=json.loads((OUT/'explanation_checks.json').read_text());assurance=[]
    for r in attacks.to_dict('records'):assurance.append(dict(control=f"Bounded attack: ±{r['temperature_budget_C']}°C / ±{r['command_budget']} commands",measure='Accepted wrong / false normal',result=f"{r['windows_with_any_accepted_wrong_class']}/40; {r['windows_with_any_accepted_false_normal']}/30 faults",limit=f"Raw attacked accuracy {r['attack_accuracy']:.1%}; review {r['worst_case_review_rate']:.1%}; 64 random attacks/window, not exhaustive"))
    assurance.extend([dict(control='Exact grouped Shapley',measure='Maximum additivity residual',result=f"{e['max_additivity_error']:.2e}",limit='20 windows, four sensor groups; not causal or human-usefulness validation'),dict(control='Explanation repeatability',measure='Identical repeated explanations',result=f"{e['repeatability']:.1%}",limit='Same model, input and reference; deterministic repeatability only'),dict(control='Explanation stability',measure='Top absolute group after +0.05°C',result=f"{e['top_group_stability_under_005C']:.1%}",limit='Same explained class; 20 windows, 5 per class'),dict(control='Scope protection',measure='Mandatory target review',result='100%',limit='Protection through abstention; target diagnostic performance remains weak'),dict(control='Model release governance',measure='Every-seed F1 / recall / missed-fault guards',result=f"{sum(d['passed_all_seeds'] for d in decisions)}/6 candidate families passed",limit='Selection is source development; validation reuse can overfit')])
    search=[dict(family=d['family'],version=d['version'].upper(),mean_F1=d['mean_f1'],seed_std=d['std_f1'],worst_F1=d['worst_f1'],promoted=d['passed_all_seeds']) for d in decisions]
    history=[dict(milestone='M2',implementation='Risk matrix, scope and severity',evidence='Original records preserved; controls mapped to tests'),dict(milestone='M3',implementation='Units, leakage prevention, partition and support audit',evidence='47 file hashes; 3 weather conditions, independent-case floor unmet'),dict(milestone='M4',implementation='Fresh RF + temporal V0/V1/V2, adaptation and review',evidence='Historical M4 retained separately; no inherited architecture weights'),dict(milestone='November 16',implementation='19 temporal runs / 662 epochs, Docker, SQLite, monitoring, recovery',evidence='experiments/end_to_end_20261116'),dict(milestone='November 17',implementation='7 classical diagnostics; 9 fresh hybrid runs / 450 epochs; 9 calibration fits; restored UI',evidence='experiments/improvement_20261117; browser, adversarial and Shapley evidence')]
    write_json(OUT/'presentation.json',dict(status='The improvement search did not clear every-seed safety gates. Retained models are shown honestly; 69% is macro F1, not accuracy. Source/target evidence is descriptive, not independent field validation.',comparison=comparison,class_metrics=classes,search=search,assurance=assurance,history=history))
    print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
