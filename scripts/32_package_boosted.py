"""Package guarded refinements and evaluate frozen post-selection partitions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from catboost import CatBoostClassifier
from aeris.modeling import load_hybrid,metrics,CLASS_NAMES
from aeris.boosted import BoostedForest
from aeris.workflow import write_bundle,SETTINGS
OUT=ROOT/'experiments/operator_redesign_20261117';W=ROOT/'data/aeris_windows_v1'
def write(p,x):p.write_text(json.dumps(x,indent=2))
def main():
    torch.set_num_threads(2);decision=json.loads((OUT/'fusion_decisions.json').read_text());assert all(v['passed_all_seeds'] for v in decision['versions'].values())
    selection={'bundle':'boosted_v2_s42','per_architecture':{v:f'boosted_{v}_s42' for v in ['v0','v1','v2']},'scope':'Source-only constrained refinement. Three booster seeds on fixed, distinct 50-epoch RF+TCN parents; not independent neural retraining. Existing target labels retrospective.','selection_frozen_before_postselection_evaluation':True};write(OUT/'release_selection.json',selection)
    comparison=[];classes=[];meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id')
    for v,name in selection['per_architecture'].items():
        r=json.loads((OUT/f'fusion_{v}_s42.json').read_text());parent=ROOT/'data/aeris_models'/f'e2e_anchor_{v}_s42';destination=ROOT/'data/aeris_models'/name
        if not destination.exists():
            model=load_hybrid(parent);cat=CatBoostClassifier();cat.load_model(str(OUT/r['booster']/'model.cbm'));model.forest=BoostedForest(model.forest,cat)
            model.configuration.update(run_name=name,parent_bundle=parent.name,learning_stage='source_boosted_refinement',boosting=dict(weight=r['weight'],class_factors=[r['normal_factor'],1,1,1],temperature=r['booster_temperature'],hybrid_temperature=r['fusion_temperature'],representation='physical_plus_full_sequence',algorithm='CatBoost 1.2.10',seed=42))
            write_bundle(destination,model,parent,{'experiment':'operator_redesign_20261117'})
        model=load_hybrid(destination);model.forest.n_jobs=1
        for role in ['source_validation','source_test','target_test']:
            p=dict(np.load(W/f'{role}.npz'));rows=meta.loc[p['window_id']];y=p.get('y')
            if y is None:y=pd.read_csv(W/'evaluation/target_window_reference.csv.gz').set_index('window_id').loc[p['window_id'],'class_id'].to_numpy('int64')
            pred=model.predict(p['X_seq'],p['X_tab']);m=metrics(y,pred['probabilities']);review=pred['needs_review']|(np.abs(p['X_seq'])>6).any((1,2))
            if role=='target_test':review[:]=True
            else:review|=np.array([int(k) not in SETTINGS[s] for k,s in zip(pred['class_id'],rows.season)])
            m.update(bundle=name,role=role,fault_to_normal=int(((y!=0)&(pred['class_id']==0)).sum()),accepted_false_normal=int(((y!=0)&(pred['class_id']==0)&~review).sum()),review_rate=float(review.mean()),accepted_windows=int((~review).sum()),accepted_accuracy=float((pred['class_id'][~review]==y[~review]).mean()) if (~review).any() else None)
            if role=='source_validation':
                expected=np.load(OUT/f'fusion_{v}_s42.npz')['probabilities'];assert np.allclose(expected,pred['probabilities'],atol=1e-10);assert abs(m['macro_f1']-r['macro_f1'])<1e-10
                classes.extend(dict(version=v.upper()+' refined',condition=c,**{k:m['per_class'][c][k] for k in ['precision','recall','f1-score','support']}) for c in CLASS_NAMES)
            write(OUT/f'{name}_{role}_metrics.json',m);comparison.append(dict(version=v.upper()+' refined',partition=role,n=len(y),accuracy=m['accuracy'],macro_F1=m['macro_f1'],macro_precision=m['per_class']['macro avg']['precision'],macro_recall=m['per_class']['macro avg']['recall'],ECE=m['ece_10_bins'],Brier=m['multiclass_brier'],review_rate=m['review_rate']));print(name,role,m['macro_f1'],m['fault_to_normal'],flush=True)
    old=json.loads((ROOT/'experiments/improvement_20261117/presentation.json').read_text());data=json.loads((OUT/'dataset_support.json').read_text());old.update(status='All three versions now have a boosted refinement that passed three booster-seed source-development checks. Gains are modest, not a high-accuracy or field-validation claim. Prior and refined results are shown on the same partitions.',comparison=comparison+old['comparison'],class_metrics=classes+old['class_metrics'],dataset_support=data['settings'],class_support=data['classes'],assurance=[dict(control='Refined model assurance',measure='Current release',result='Source development guards passed; additional release checks in progress',limit='Earlier assurance rows below describe the prior baseline until replaced by the current measured report')]+old['assurance']);write(OUT/'presentation.json',old)
    write(OUT/'release_catalog.json',{'models':[dict(version=v,bundle=n,stage='Boosted refinement') for v,n in selection['per_architecture'].items()]+[dict(version=v,bundle=f'e2e_anchor_{v}_s42',stage='Prior baseline') for v in ['v0','v1','v2']]})
if __name__=='__main__':main()
