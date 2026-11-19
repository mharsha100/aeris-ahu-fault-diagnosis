"""Separate source-only information study; does not change the deployed input contract."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from aeris.physical import PhysicalFeatures
from aeris.modeling import metrics
OUT=ROOT/'experiments/operator_redesign_20261117';W=ROOT/'data/aeris_windows_v1'
COLUMNS=['AHU Supply Air Temperature Setpoint (°K)','Supply Air Mass Flow Rate (kg/s)','Outdoor Air Mass Flow Rate (kg/s)','Return Air Mass Flow Rate in the Mixing Box (kg/s)','Measured AHU Supply Air Duct Static Pressure (Pa)','AHU Supply Air Duct Static Pressure Setpoint (Pa)','AHU Outdoor Air Wet Bulb Temperature (°K)']
def write(p,x):p.write_text(json.dumps(x,indent=2))
def main():
    plan={'purpose':'Test whether richer source sensor context contains diagnostic information missing from the 10-channel common contract. Separate source-only experiment, not a replacement or transfer claim.','channels':COLUMNS,'excluded':'true positions, labels, scenario identifiers, dates and time values as model inputs','alignment':'Original window IDs, row ranges and train/validation partitions unchanged. File identities only join rows; never features.','limitations':'Simulated source channels; sensor availability, measurement quality and semantics at target sites are not established. Any larger gain is not comparable as an algorithm-only improvement. No deployment from this script.','runs':2,'seeds':[42,43],'iterations':1200}
    if not (OUT/'sensor_context_plan.json').exists():write(OUT/'sensor_context_plan.json',plan)
    meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id');manifest=pd.read_csv(ROOT/'data/aeris_prepared/input_manifest.csv');files={hashlib.sha256(r.relative_path.encode()).hexdigest()[:16]:ROOT/'data'/r.relative_path for r in manifest.itertuples() if r.domain=='source'};cache={};packs={r:dict(np.load(W/f'source_{r}.npz')) for r in ['train','validation']};X={}
    for role,p in packs.items():
        rows=meta.loc[p['window_id']];extra=np.empty((len(rows),15,len(COLUMNS)),np.float32)
        for fid,indices in rows.groupby(rows.start_row_id.str.split(':').str[0]).indices.items():
            if fid not in cache:
                frame=pd.read_csv(files[fid],usecols=lambda c:c.strip() in COLUMNS);frame.columns=frame.columns.str.strip();a=frame[COLUMNS].to_numpy(np.float32);a[:,[0,6]]-=273.15;cache[fid]=a
            group=rows.iloc[indices];starts=group.start_row_id.str.split(':').str[-1].astype(int).to_numpy();ends=group.end_row_id.str.split(':').str[-1].astype(int).to_numpy();assert np.all(ends-starts==14);extra[indices]=cache[fid][starts[:,None]+np.arange(15)]
        assert np.isfinite(extra).all()
        stats=np.concatenate([extra.mean(1),extra.std(1),extra.min(1),extra.max(1),extra[:,-1],(extra[:,-1]-extra[:,0])/14],1)
        X[role]=np.concatenate([PhysicalFeatures().transform(p['X_tab']),p['X_seq'].reshape(len(rows),-1),stats],1)
    for seed in [42,43]:
        dest=OUT/f'sensor_context_s{seed}';dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        start=time.perf_counter();model=CatBoostClassifier(iterations=1200,depth=8,learning_rate=.05,l2_leaf_reg=5,loss_function='MultiClass',eval_metric='TotalF1:average=Macro;use_weights=false',auto_class_weights='SqrtBalanced',random_seed=seed,task_type='GPU',devices='0',thread_count=4,border_count=128,allow_writing_files=False,verbose=300)
        model.fit(X['train'],packs['train']['y'],eval_set=(X['validation'],packs['validation']['y']),use_best_model=True);p=model.predict_proba(X['validation']);r=metrics(packs['validation']['y'],p);r.update(seed=seed,seconds=time.perf_counter()-start,features=X['train'].shape[1],additional_channels=COLUMNS,selected_iteration=model.get_best_iteration()+1,iterations_completed=len(model.evals_result_['learn']['MultiClass']),fault_to_normal=int(((packs['validation']['y']!=0)&(p.argmax(1)==0)).sum()),deployed=False);write(dest/'result.json',r);write(dest/'curves.json',model.evals_result_);model.save_model(str(dest/'model.cbm'));print('SENSOR CONTEXT',seed,r['macro_f1'],r['fault_to_normal'],flush=True)
if __name__=='__main__':main()
