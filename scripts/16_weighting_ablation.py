"""Isolate RF class weighting, reusing the matched temporal branch without retraining."""
from pathlib import Path
import sys,time,json,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from sklearn.ensemble import RandomForestClassifier
from aeris.modeling import load_hybrid,fit_temperature,temperature_probabilities,metrics
from aeris.training import write_json
OUT=ROOT/'experiments/end_to_end_20261116';W=ROOT/'data/aeris_windows_v1'
def main():
    torch.set_num_threads(2);model=load_hybrid(ROOT/'data/aeris_models/e2e_anchor_v0_s42');model.forest.n_jobs=2
    train,val,cal=[dict(np.load(W/f'{r}.npz')) for r in ['source_train','source_validation','source_calibration']]
    pv=model.predict(val['X_seq'],val['X_tab']);pc=model.predict(cal['X_seq'],cal['X_tab']);rows=[]
    for weight in ['balanced_subsample',None]:
        start=time.perf_counter();rf=model.forest if weight else RandomForestClassifier(**{**model.forest.get_params(),'class_weight':None,'n_jobs':2})
        if weight is None:rf.fit(train['X_tab'],train['y'])
        fv,fc=rf.predict_proba(val['X_tab']),rf.predict_proba(cal['X_tab']);ft=fit_temperature(fc,cal['y']);fv=temperature_probabilities(fv,ft);fc=temperature_probabilities(fc,ft)
        w=model.configuration['forest_weight'];mix=w*fv+(1-w)*pv['temporal_probabilities'];ht=fit_temperature(w*fc+(1-w)*pc['temporal_probabilities'],cal['y']);mix=temperature_probabilities(mix,ht)
        mm=metrics(val['y'],mix);pred=mix.argmax(1);review=(mix.max(1)<.9)|(fv.argmax(1)!=pv['temporal_probabilities'].argmax(1))
        rows.append({'class_weight':weight,'model':'v0','seed':42,'fixed_fusion_weight':w,'seconds':time.perf_counter()-start,'rf_reused':weight is not None,'temporal_reused':True,'macro_f1':mm['macro_f1'],'fault_to_normal':int(((val['y']!=0)&(pred==0)).sum()),'normal_false_alarm':int(((val['y']==0)&(pred!=0)).sum()),'per_class':mm['per_class'],'brier':mm['multiclass_brier'],'review_fraction':float(review.mean()),'accepted_false_normal_model_gate':int(((val['y']!=0)&(pred==0)&~review).sum())})
        write_json(OUT/'weighting_ablation.json',{'hypothesis':'Normal is rare in the source; test whether existing balancing improves diagnostic tradeoffs. All other RF and temporal parameters fixed; calibration refit only on source calibration.','rows':rows,'promotion':'Diagnostic experiment only; does not bypass predeclared three-seed guard.'})
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
