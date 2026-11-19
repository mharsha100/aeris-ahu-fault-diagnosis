"""Shared physical representation improvement; fresh temporal models, frozen split."""
from pathlib import Path
import sys,json,time,copy,random
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch,joblib,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from aeris.modeling import *
from aeris.physical import PhysicalForest
from aeris.training import train_temporal,write_json
from aeris.workflow import write_bundle
OUT=ROOT/'experiments/improvement_20261117';W=ROOT/'data/aeris_windows_v1'
KINDS={'v0':'causal_tcn_last','v1':'causal_tcn_attention','v2':'causal_tcn_combined'}
def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    train,val,cal=[dict(np.load(W/f'source_{k}.npz')) for k in ['train','validation','calibration']]
    scaler=json.loads((W/'source_scaler.json').read_text());protocol=json.loads((W/'window_protocol.json').read_text())
    write_json(OUT/'hybrid_plan.json',{'runs':9,'seeds':[42,43,44],'epochs':50,'fresh_temporal':True,'same_encoder':'32 channels, dropout .15, dilations 1/2/4, weighted loss, lr .001','change':'Common physical context residual head and deterministic physical RF features','target_used':False,'selection':'Macro F1 >= matched anchor +.01; no class recall decrease >.03; no increased fault-normal or accepted false-normal counts, every seed. Otherwise retain parent.'})
    for seed in [42,43,44]:
        fp=OUT/f'physical_forest_s{seed}.joblib'
        if fp.exists():forest=joblib.load(fp)
        elif seed==42:forest=joblib.load(OUT/'rf_physics_bal/estimator.joblib');joblib.dump(forest,fp,compress=3)
        else:
            forest=PhysicalForest(RandomForestClassifier(n_estimators=240,max_depth=None,min_samples_leaf=2,max_features=.8,class_weight='balanced_subsample',n_jobs=4,random_state=seed));forest.fit(train['X_tab'],train['y']);joblib.dump(forest,fp,compress=3)
        forest.n_jobs=2;fv,fc=[forest.predict_proba(p['X_tab']) for p in [val,cal]];ft=fit_temperature(fc,cal['y'])
        for version,kind in KINDS.items():
            name=f'physics_{version}_s{seed}';dest=OUT/name;dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():continue
            start=time.perf_counter();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            architecture=dict(input_features=10,channels=32,dropout=.15,dilations=[1,2,4],classes=4,raw_mean=scaler['mean'],raw_scale=scaler['scale'],context_width=64)
            write_json(dest/'config.json',dict(name=name,seed=seed,epochs=50,architecture=architecture,kind='physical_'+kind,forest='rf_physics_bal parameters',device='cuda',source_train_sha256=sha256_file(W/'source_train.npz')))
            nn=create_temporal('physical_'+kind,architecture).to('cuda');history,best=train_temporal(nn,train,val,device='cuda',epochs=50,batch_size=256,patience=0,seed=seed,checkpoint_dir=dest)
            tv,tc=[softmax(neural_logits(nn,p['X_seq'],'cuda')) for p in [val,cal]];tt=fit_temperature(tc,cal['y']);sweep=[]
            for weight in [.25,.5,.65,.75,.85,.9]:
                pp=weight*temperature_probabilities(fv,ft)+(1-weight)*temperature_probabilities(tv,tt);mm=metrics(val['y'],pp);sweep.append(dict(weight=weight,macro_f1=mm['macro_f1'],fault_to_normal=int(((val['y']!=0)&(pp.argmax(1)==0)).sum())))
            chosen=max(sweep,key=lambda r:(r['macro_f1'],-r['fault_to_normal']));weight=chosen['weight'];ht=fit_temperature(weight*temperature_probabilities(fc,ft)+(1-weight)*temperature_probabilities(tc,tt),cal['y'])
            cfg=dict(artifact_version='aeris-hybrid-v1',model_version=version,temporal_kind='physical_'+kind,temporal_architecture=architecture,classes=CLASS_NAMES,learning_stage='source_physical_representation',forest_weight=weight,temporal_weight=1-weight,temperatures=dict(forest=ft,temporal=tt,hybrid=ht),review_confidence_threshold=.9,selected_temporal_epoch=best,seed=seed,target_data_used=False,test_data_used=False,run_name=name)
            model=HybridPredictor(forest,nn,cfg,scaler,protocol,'cuda');pred=model.predict(val['X_seq'],val['X_tab']);m=metrics(val['y'],pred['probabilities']);m.update(name=name,version=version,seed=seed,epochs=50,best_epoch=best,seconds=time.perf_counter()-start,fusion=sweep,fault_to_normal=int(((val['y']!=0)&(pred['class_id']==0)).sum()))
            np.savez_compressed(dest/'predictions.npz',**pred);write_json(dest/'pending_result.json',m);restored=write_bundle(ROOT/'data/aeris_models'/name,model,W,{'experiment':'improvement_20261117'})
            restored.forest.n_jobs=1;cpu=restored.predict(val['X_seq'],val['X_tab']);m['cpu_metrics']=metrics(val['y'],cpu['probabilities']);m['cpu_class_disagreements']=int((cpu['class_id']!=pred['class_id']).sum());write_json(dest/'result.json',m)
            print('COMPLETE',name,m['macro_f1'],m['fault_to_normal'],flush=True)
    write_json(OUT/'hybrids_complete.json',{'runs':9,'epochs':450})
if __name__=='__main__':main()
