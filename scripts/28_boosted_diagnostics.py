"""Stronger tabular and full-window diagnostics on the unchanged source split."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,__version__
from aeris.physical import PhysicalFeatures
from aeris.modeling import metrics,sha256_file,CLASS_NAMES
OUT=ROOT/'experiments/operator_redesign_20261117';W=ROOT/'data/aeris_windows_v1'
def write(path,x):path.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True);packs={k:dict(np.load(W/f'source_{k}.npz')) for k in ['train','validation','calibration']}
    plan={'created_before_results':True,'method':'CatBoost 1.2.10; three representations; no target data or metadata features','representations':['66 summaries','112 physical features','112 physical features + all 150 standardized sequential values'],'screening':8,'iterations_each':1200,'early_stopping':False,'checkpoint_selection':'source validation macro F1; all 1200 iterations run','seed':42,'promotion':'Screening alone cannot promote. Require every seed42/43/44 of a hybrid retaining RF and temporal branches to improve matched-anchor macro F1 by >=.01, no recall loss >.03, no increase in fault-normal or accepted false-normal. Target never used for selection.','versions':'Existing V0/V1/V2 parents remain distinct; any successful boost is a same-architecture improvement stage.'}
    if not (OUT/'boost_plan.json').exists():write(OUT/'boost_plan.json',plan)
    meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id');support=[];classes=[]
    for role,pack in packs.items():
        rows=meta.loc[pack['window_id']]
        for (dataset,season),n in rows.groupby(['dataset','season']).size().items():support.append(dict(partition='source_'+role,dataset=dataset,setting=season,windows=int(n)))
        for k,name in enumerate(CLASS_NAMES):classes.append(dict(partition='source_'+role,condition=name,windows=int((pack['y']==k).sum()),share=float((pack['y']==k).mean())))
    write(OUT/'dataset_support.json',dict(settings=support,classes=classes,interpretation='Prepared dataset support, not live inference traffic. Source days share fault recordings and weather groups; windows are not independent sites/cases.'))
    X={}
    for role,p in packs.items():
        phys=PhysicalFeatures().transform(p['X_tab']);X[role]={'base':p['X_tab'],'physics':phys,'sequence':np.concatenate([phys,p['X_seq'].reshape(len(phys),-1)],axis=1)}
    jobs=[(v,d,'Balanced') for v in ['base','physics','sequence'] for d in [6,8]]+[(v,8,'SqrtBalanced') for v in ['physics','sequence']]
    for view,depth,balance in jobs:
        name=f'cat_{view}_d{depth}_{balance.lower()}_s42';dest=OUT/name;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        cfg=dict(iterations=1200,depth=depth,learning_rate=.05,l2_leaf_reg=5,loss_function='MultiClass',eval_metric='TotalF1:average=Macro;use_weights=false',auto_class_weights=balance,random_seed=42,task_type='GPU',devices='0',thread_count=4,border_count=128,allow_writing_files=False,verbose=200)
        write(dest/'config.json',dict(parameters=cfg,view=view,catboost_version=__version__,source_train_sha256=sha256_file(W/'source_train.npz')))
        start=time.perf_counter();model=CatBoostClassifier(**cfg);model.fit(X['train'][view],packs['train']['y'],eval_set=(X['validation'][view],packs['validation']['y']),use_best_model=True)
        preds={role:model.predict_proba(X[role][view]) for role in packs};r=metrics(packs['validation']['y'],preds['validation']);r.update(name=name,view=view,depth=depth,balance=balance,iterations_completed=len(model.evals_result_['learn']['MultiClass']),selected_iteration=model.get_best_iteration()+1,seconds=time.perf_counter()-start,fault_to_normal=int(((packs['validation']['y']!=0)&(preds['validation'].argmax(1)==0)).sum()),training_metrics=metrics(packs['train']['y'],preds['train']))
        assert r['iterations_completed']==1200;model.save_model(str(dest/'model.cbm'));np.savez_compressed(dest/'predictions.npz',**preds);write(dest/'curves.json',model.evals_result_);write(dest/'result.json',r);print('COMPLETE',name,r['macro_f1'],r['fault_to_normal'],r['seconds'],flush=True)
    print('BOOSTED SCREEN COMPLETE',flush=True)
if __name__=='__main__':main()
