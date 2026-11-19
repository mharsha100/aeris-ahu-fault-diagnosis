"""Source-only representation/algorithm diagnostics; no target score selection."""
from pathlib import Path
import os,sys,json,time
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from aeris.physical import PhysicalFeatures,PhysicalForest
from aeris.modeling import metrics
OUT=ROOT/'experiments/improvement_20261117';OUT.mkdir(exist_ok=True)
def main():
    packs={k:dict(np.load(ROOT/'data/aeris_windows_v1'/f'source_{k}.npz')) for k in ['train','validation','calibration']}
    plan={'created_before_results':True,'selection_inputs':'frozen source train/validation/calibration only','max_diagnostic_trials':8,'max_seconds':7200,'hypotheses':['Temperature/command interaction ratios make physical faults easier for axis-aligned trees.','Compare boosted trees to diagnose RF underfitting; RF+TNN remains required for deployment.','Do not use dates, labels, simulated true positions or file identity as inputs.'],'promotion':'Macro F1 must improve by >=0.01 over matching Nov16 anchor; no increase in fault-normal or accepted false-normal counts; no per-class recall decrease >.03. Confirm three seeds for each architecture. All target results retrospective and only after selection.'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    jobs=[('rf_physics_bal',PhysicalForest(RandomForestClassifier(n_estimators=240,max_depth=None,min_samples_leaf=2,max_features=.8,class_weight='balanced_subsample',n_jobs=4,random_state=42))),('rf_physics_none',PhysicalForest(RandomForestClassifier(n_estimators=240,max_depth=None,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42))),('extra_physics',PhysicalForest(ExtraTreesClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,max_features=.8,class_weight='balanced',n_jobs=4,random_state=42)))]
    for features in ['base','physics']:
        for leaf in [15,31]:jobs.append((f'hgb_{features}_{leaf}',HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=leaf,learning_rate=.07,l2_regularization=2,min_samples_leaf=30,early_stopping=False,class_weight='balanced',random_state=42)))
    for name,model in jobs:
        dest=OUT/name;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        start=time.perf_counter();X={k:p['X_tab'] for k,p in packs.items()}
        if name.startswith('hgb_physics'):X={k:PhysicalFeatures().transform(x) for k,x in X.items()}
        model.fit(X['train'],packs['train']['y']);prob=model.predict_proba(X['validation']);m=metrics(packs['validation']['y'],prob)
        m.update(name=name,seconds=time.perf_counter()-start,fault_to_normal=int(((packs['validation']['y']!=0)&(prob.argmax(1)==0)).sum()),parameters=model.get_params(deep=True).__repr__())
        (dest/'result.json').write_text(json.dumps(m,indent=2));joblib.dump(model,dest/'estimator.joblib',compress=3)
        np.savez_compressed(dest/'predictions.npz',validation=prob,calibration=model.predict_proba(X['calibration']))
        print(name,m['macro_f1'],m['fault_to_normal'],m['seconds'],flush=True)
    print('DIAGNOSTIC SEARCH COMPLETE',flush=True)
if __name__=='__main__':main()
