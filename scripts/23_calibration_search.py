"""Fit fusion/class-decision factors on source calibration, then evaluate development."""
from pathlib import Path
import sys,json,copy,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from aeris.modeling import load_hybrid,metrics,fit_temperature,temperature_probabilities
from aeris.workflow import write_bundle
from aeris.training import write_json
OUT=ROOT/'experiments/improvement_20261117';W=ROOT/'data/aeris_windows_v1'
def fast_score(y,p):
    c=np.bincount(y*4+p.argmax(1),minlength=16).reshape(4,4);tp=c.diagonal();return float(np.mean(2*tp/np.maximum(c.sum(0)+c.sum(1),1))),int(c[1:,0].sum())
def main():
    torch.set_num_threads(2);cal=dict(np.load(W/'source_calibration.npz'));val=dict(np.load(W/'source_validation.npz'))
    write_json(OUT/'class_decision_plan.json',{'hypothesis':'Normal is rare; calibrating class decision balance may improve macro F1 without unsafe missed-fault increases.','fit_partition':'source_calibration only; no source validation used to fit these factors','normal_factors':[.6,.8,1,1.2,1.4,1.6,2],'cooling_factors':[.85,1,1.15],'RF_weights':[.25,.5,.65,.75,.85,.9],'validation_guard':'Same per-class recall, false-normal and three-seed gates as the physical family. No target fitting.'})
    for rp in sorted(OUT.glob('physics_*/result.json')):
        original=json.loads(rp.read_text());name='calibrated_'+original['name'];dest=OUT/name;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        start=time.perf_counter();parent=ROOT/'data/aeris_models'/original['name'];m=load_hybrid(parent);m.forest.n_jobs=2;p=m.predict(cal['X_seq'],cal['X_tab']);candidates=[]
        for w in [.25,.5,.65,.75,.85,.9]:
            base=w*p['forest_probabilities']+(1-w)*p['temporal_probabilities']
            for normal in [.6,.8,1,1.2,1.4,1.6,2]:
                for cooling in [.85,1,1.15]:
                    factors=[normal,1,cooling,1];q=base*factors;q/=q.sum(1,keepdims=True);f1,miss=fast_score(cal['y'],q);candidates.append(dict(weight=w,factors=factors,calibration_f1=f1,calibration_false_normal=miss))
        chosen=max(candidates,key=lambda x:(x['calibration_f1'],-x['calibration_false_normal']));w=chosen['weight'];q=(w*p['forest_probabilities']+(1-w)*p['temporal_probabilities'])*chosen['factors'];q/=q.sum(1,keepdims=True)
        m.configuration.update(forest_weight=w,temporal_weight=1-w,class_factors=chosen['factors'],parent_bundle=original['name'],learning_stage='source_physics_calibrated_decisions',class_factor_fit_partition='source_calibration');m.configuration['temperatures']['hybrid']=fit_temperature(q,cal['y'])
        pred=m.predict(val['X_seq'],val['X_tab']);result=metrics(val['y'],pred['probabilities']);result.update(name=name,version=original['version'],seed=original['seed'],seconds=time.perf_counter()-start,chosen=chosen,candidates=candidates,fault_to_normal=int(((val['y']!=0)&(pred['class_id']==0)).sum()))
        write_json(dest/'pending_result.json',result);np.savez_compressed(dest/'predictions.npz',**pred);write_bundle(ROOT/'data/aeris_models'/name,m,parent,{'experiment':'improvement_20261117'});write_json(dest/'result.json',result);print(name,result['macro_f1'],result['fault_to_normal'],chosen,flush=True)
if __name__=='__main__':main()
