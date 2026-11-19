from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from aeris.modeling import load_hybrid
from aeris.workflow import canonical_window,diagnose
from aeris.training import write_json
OUT=ROOT/'experiments/end_to_end_20261116';W=ROOT/'data/aeris_windows_v1'
def main():
    torch.set_num_threads(2);p=dict(np.load(W/'source_validation.npz'));meta=pd.read_csv(W/'window_metadata.csv.gz').set_index('window_id').loc[p['window_id']].reset_index();rng=np.random.default_rng(20261116);indices=np.array([rng.choice(np.flatnonzero(p['y']==k),25,replace=False) for k in range(4)]).T.reshape(-1);rows=[]
    for name in ['v0','v1_attention']:
        m=load_hybrid(ROOT/'data/aeris_models'/name);m.forest.n_jobs=1;rng=np.random.default_rng(20261116)
        for i in indices:
            raw=canonical_window(p,int(i),m.scaler);cases={'clean':raw.copy(),'noise_0.2C':raw.copy(),'offset_supply_1C':raw.copy(),'dropout':raw.copy(),'stuck_supply':raw.copy(),'unknown_setting':raw.copy()};cases['noise_0.2C'][:,:4]+=np.clip(rng.normal(0,.2,(15,4)),-.6,.6);cases['offset_supply_1C'][:,0]+=1;cases['dropout'][7,0]=np.nan;cases['stuck_supply'][:,0]=raw[0,0]
            for kind,data in cases.items():
                r=diagnose(m,data,setting='Unknown' if kind=='unknown_setting' else meta.iloc[i].season,timestamps=np.arange(15)*60,recorded_source=kind!='unknown_setting');rows.append(dict(bundle=name,window_id=p['window_id'][i],case=kind,true_class=int(p['y'][i]),predicted=r.get('class_id'),status=r['status'],false_normal=bool(p['y'][i]!=0 and r.get('class_id')==0),accepted_false_normal=bool(p['y'][i]!=0 and r.get('class_id')==0 and r['status']=='research_prediction')))
        pd.DataFrame(rows).to_csv(OUT/'baseline_robustness_cases.csv',index=False)
    df=pd.DataFrame(rows);df.groupby(['bundle','case']).agg(windows=('window_id','size'),false_normal=('false_normal','sum'),accepted_false_normal=('accepted_false_normal','sum')).to_csv(OUT/'baseline_robustness_summary.csv');print('Completed 1,200 baseline hybrid corruption/control requests.')
if __name__=='__main__':main()
