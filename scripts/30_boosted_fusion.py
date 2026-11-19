"""Source-only constrained RF+TCN+boosted-tree refinement, with held seed checks."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from catboost import CatBoostClassifier
from aeris.modeling import load_hybrid,metrics,fit_temperature,temperature_probabilities,CLASS_NAMES
from aeris.physical import PhysicalFeatures
OUT=ROOT/'experiments/operator_redesign_20261117';W=ROOT/'data/aeris_windows_v1';OLD=ROOT/'experiments/improvement_20261117'
def write(p,x):p.write_text(json.dumps(x,indent=2))
def quick(y,p):
    c=np.bincount(y*4+p.argmax(1),minlength=16).reshape(4,4);tp=c.diagonal();return float(np.mean(2*tp/np.maximum(c.sum(0)+c.sum(1),1))),c[1:,0].sum(),tp/np.maximum(c.sum(1),1)
def main():
    torch.set_num_threads(2);packs={k:dict(np.load(W/f'source_{k}.npz')) for k in ['train','validation','calibration']};y=packs['validation']['y']
    candidates=[json.loads(p.read_text()) for p in OUT.glob('cat_*/result.json')];assert len(candidates)==8
    winner=max(candidates,key=lambda r:r['macro_f1']);parentdir=OUT/winner['name'];cfg=json.loads((parentdir/'config.json').read_text());view=winner['view']
    plan=dict(parent_booster=winner['name'],screening_seed=42,confirmation_seeds=[43,44],booster_weights=[.25,.5,.65,.8,.9],normal_factors=[.8,1.,1.2,1.4,1.6,1.8,2.],selection='Constrained source-development hyperparameter selection; fixed factors/weight for seed43/44. Scalar confidence temperatures fitted on calibration only. No target selection.',guard='Each seed: gain >=.01 macro F1, no recall drop >.03, no increased fault-normal or accepted false-normal. Existing distinct 50-epoch RF+TCN parents fixed; seed replication concerns the new booster, not independent neural retraining.')
    if not (OUT/'fusion_plan.json').exists():write(OUT/'fusion_plan.json',plan)
    X={}
    for role,p in packs.items():
        phys=PhysicalFeatures().transform(p['X_tab']);X[role]={'base':p['X_tab'],'physics':phys,'sequence':np.concatenate([phys,p['X_seq'].reshape(len(phys),-1)],1)}[view]
    parent={};anchors={}
    for version in ['v0','v1','v2']:
        name=f'e2e_anchor_{version}_s42';m=load_hybrid(ROOT/'data/aeris_models'/name);m.forest.n_jobs=1;parent[version]={role:m.predict(packs[role]['X_seq'],packs[role]['X_tab']) for role in ['validation','calibration']};anchors[version]=json.loads((OLD/(name+'_release_metrics.json')).read_text())
    decisions={};results=[]
    for seed in [42,43,44]:
        if seed==42:boost=dict(np.load(parentdir/'predictions.npz'))
        else:
            folder=OUT/f'confirm_booster_s{seed}';folder.mkdir(exist_ok=True);params=cfg['parameters'].copy();params['random_seed']=seed
            if (folder/'predictions.npz').exists():boost=dict(np.load(folder/'predictions.npz'))
            else:
                model=CatBoostClassifier(**params);model.fit(X['train'],packs['train']['y'],eval_set=(X['validation'],y),use_best_model=True);model.save_model(str(folder/'model.cbm'));boost={r:model.predict_proba(X[r]) for r in ['validation','calibration']};np.savez_compressed(folder/'predictions.npz',**boost);write(folder/'training.json',dict(seed=seed,iterations_completed=len(model.evals_result_['learn']['MultiClass']),selected_iteration=model.get_best_iteration()+1,parameters=params,representation=view));write(folder/'curves.json',model.evals_result_)
        bt=fit_temperature(boost['calibration'],packs['calibration']['y']);bp={r:temperature_probabilities(boost[r],bt) for r in ['validation','calibration']}
        for version,pp in parent.items():
            anchor=anchors[version];ar=np.array([anchor['per_class'][c]['recall'] for c in CLASS_NAMES]);trials=[]
            if seed==42:
                for w in plan['booster_weights']:
                    for factor in plan['normal_factors']:
                        p=((1-w)*pp['validation']['probabilities']+w*bp['validation'])*[factor,1,1,1];p/=p.sum(1,keepdims=True);f1,miss,recall=quick(y,p);passed=f1>=anchor['macro_f1']+.01 and miss<=anchor['fault_to_normal'] and (ar-recall).max()<=.03
                        trials.append(dict(weight=w,normal_factor=factor,macro_f1=f1,fault_to_normal=int(miss),max_recall_drop=float((ar-recall).max()),passed=bool(passed)))
                qualified=[r for r in trials if r['passed']];chosen=max(qualified or trials,key=lambda r:r['macro_f1']);decisions[version]=chosen;write(OUT/f'fusion_screen_{version}.json',dict(chosen=chosen,trials=trials))
            chosen=decisions[version];w=chosen['weight'];factor=chosen['normal_factor'];prob={}
            for role in ['validation','calibration']:
                q=((1-w)*pp[role]['probabilities']+w*bp[role])*[factor,1,1,1];prob[role]=q/q.sum(1,keepdims=True)
            ht=fit_temperature(prob['calibration'],packs['calibration']['y']);p=temperature_probabilities(prob['validation'],ht);r=metrics(y,p);f1,miss,recall=quick(y,p)
            # Conservative added-branch agreement gate plus source-support gate.
            pred=p.argmax(1);disagreement=pp['validation']['branch_disagreement']|(boost['validation'].argmax(1)!=pp['validation']['temporal_probabilities'].argmax(1))|(pred!=pp['validation']['temporal_probabilities'].argmax(1));review=(p.max(1)<.9)|disagreement|(np.abs(packs['validation']['X_seq'])>6).any((1,2));accepted_fn=int(((y!=0)&(pred==0)&~review).sum());passed=f1>=anchor['macro_f1']+.01 and miss<=anchor['fault_to_normal'] and (ar-recall).max()<=.03 and accepted_fn<=anchor['accepted_false_normal']
            r.update(version=version,seed=seed,weight=w,normal_factor=factor,booster_temperature=bt,fusion_temperature=ht,fault_to_normal=int(miss),max_recall_drop=float((ar-recall).max()),accepted_false_normal=accepted_fn,passed=bool(passed),booster=winner['name'] if seed==42 else f'confirm_booster_s{seed}',view=view);write(OUT/f'fusion_{version}_s{seed}.json',r);np.savez_compressed(OUT/f'fusion_{version}_s{seed}.npz',probabilities=p,review=review);results.append(r);print('FUSION',version,seed,f1,miss,passed,flush=True)
    write(OUT/'fusion_decisions.json',dict(versions={v:dict(passed_all_seeds=all(r['passed'] for r in results if r['version']==v),chosen=decisions[v]) for v in parent},results=results))
if __name__=='__main__':main()
