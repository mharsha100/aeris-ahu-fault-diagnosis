"""Reproducible ART HopSkipJump attack and SHAP complete-model evaluation."""
from pathlib import Path
import sys,json,time,contextlib,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch,art,shap
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
from aeris.modeling import load_hybrid
from aeris.workflow import canonical_window,diagnose
from aeris.operations import grouped_explanation
from aeris.training import write_json

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--bundle-path',type=Path,default=ROOT/'data/aeris_models/boosted_v2_s42');parser.add_argument('--output',type=Path,default=ROOT/'experiments/adaptive_lifecycle_20261117');args=parser.parse_args()
 torch.set_num_threads(2);out=args.output;out.mkdir(parents=True,exist_ok=True)
 m=load_hybrid(args.bundle_path);m.forest.n_jobs=1
 w=ROOT/'data/aeris_windows_v1';pack=dict(np.load(w/'source_validation.npz'));meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id').loc[pack['window_id']].reset_index()
 rng=np.random.default_rng(20261117);ix=np.array([rng.choice(np.flatnonzero(pack['y']==k),10,replace=False) for k in range(4)]).T.reshape(-1)
 plan=dict(library='adversarial-robustness-toolbox',version=art.__version__,attack='HopSkipJump',norm='inf',windows=40,class_support=[10]*4,seed=20261117,max_iter=5,max_eval=64,init_eval=20,init_size=20,batch_size=64,model='boosted_v2_s42; source-only model now called lifecycle V0',constraints='Perturb only 4 temperatures and 4 commands, with per-input bounded coordinates. Fan, occupancy and timestamps fixed. Entire RF + temporal + CatBoost predictor queried.',limitation='Finite-budget offline source-development attack. Not a robustness certificate, target-site validation or a security scan of the application.')
 plan['model']=args.bundle_path.name
 write_json(out/'art_plan.json',plan);rows=[];expl=[];cols=[0,1,2,3,5,6,7,8]
 for case,i in enumerate(ix):
  raw=canonical_window(pack,int(i),m.scaler);truth=int(pack['y'][i]);season=meta.iloc[i].season;clean=m.predict_raw(raw);correct=int(clean['class_id'][0])==truth
  for eps,ceps in [(.2,.01),(.5,.02),(1.,.05)]:
   v=raw[:,cols].astype('float64');budget=np.array([eps]*4+[ceps]*4);lo=np.maximum(v-budget,np.array([-60]*4+[0]*4));hi=np.minimum(v+budget,np.array([100]*4+[1]*4));width=hi-lo;queries=[0]
   def decode(z):
    x=np.repeat(raw[None],len(z),axis=0);x[:,:,cols]=(lo+np.clip(z.reshape(-1,15,8),0,1)*width).astype('float32');return x
   def predict(z):
    queries[0]+=len(z);return m.predict_raw(decode(z))['probabilities'].astype('float32')
   x=((v-lo)/width).reshape(1,-1).astype('float32');adv=x.copy();started=time.perf_counter()
   if correct:
    np.random.seed(20261117+case)
    estimator=BlackBoxClassifier(predict_fn=predict,input_shape=(120,),nb_classes=4,clip_values=(0.,1.))
    attack=HopSkipJump(classifier=estimator,norm=np.inf,max_iter=5,max_eval=64,init_eval=20,init_size=20,batch_size=64,verbose=False)
    adv=attack.generate(x=x,y=np.eye(4,dtype='float32')[[truth]])
   changed=decode(adv)[0];res=diagnose(m,changed,setting=season,timestamps=np.arange(15)*60,recorded_source=True)
   if res['status']=='blocked':raise RuntimeError('ART violated physical contract')
   is_correct=res['class_id']==truth;review=res['status']=='review_required'
   assert np.max(abs(changed[:,:4]-raw[:,:4]))<=eps+1e-5
   assert np.max(abs(changed[:,5:9]-raw[:,5:9]))<=ceps+1e-5
   assert np.array_equal(changed[:,[4,9]],raw[:,[4,9]])
   rows.append(dict(window_id=str(pack['window_id'][i]),truth=truth,temperature_budget=eps,command_budget=ceps,clean_correct=correct,attacked_correct=is_correct and correct,successful_attack=correct and not is_correct,review_required=review,accepted_wrong=not is_correct and not review,accepted_false_normal=truth!=0 and res['class_id']==0 and not review,queried_windows=queries[0],seconds=time.perf_counter()-started))
  if case<20:
   a=grouped_explanation(m,raw);b=grouped_explanation(m,raw);small=raw.copy();small[:,:4]+=.05;c=grouped_explanation(m,small,class_id=int(clean['class_id'][0]))
   av=np.array([x['shapley_value'] for x in a['values']]);cv=np.array([x['shapley_value'] for x in c['values']])
   expl.append(dict(window_id=str(pack['window_id'][i]),additivity_error=a['additivity_error'],repeat_identical=a==b,top_group_stable=int(abs(av).argmax())==int(abs(cv).argmax())))
  pd.DataFrame(rows).to_csv(out/'art_cases.csv',index=False);print('ART',case+1,'/ 40',flush=True)
 d=pd.DataFrame(rows);summary=[]
 for eps,g in d.groupby('temperature_budget'):
  summary.append(dict(temperature_budget_C=eps,command_budget=float(g.command_budget.iloc[0]),windows=len(g),clean_accuracy=float(g.clean_correct.mean()),attack_accuracy=float(g.attacked_correct.mean()),clean_correct=int(g.clean_correct.sum()),attack_successes=int(g.successful_attack.sum()),attack_success_rate=float(g.successful_attack.sum()/g.clean_correct.sum()),accepted_wrong=int(g.accepted_wrong.sum()),accepted_false_normal=int(g.accepted_false_normal.sum()),review_rate=float(g.review_required.mean()),queried_windows=int(g.queried_windows.sum())))
 pd.DataFrame(summary).to_csv(out/'art_summary.csv',index=False)
 write_json(out/'shap_checks.json',dict(library='shap',version=shap.__version__,explainer='ExactExplainer',scope='Four sensor-group contributions to the full predictor probability; 16 coalitions.',cases=expl,max_additivity_error=max(x['additivity_error'] for x in expl),repeatability=float(np.mean([x['repeat_identical'] for x in expl])),top_group_stability=float(np.mean([x['top_group_stable'] for x in expl])),limitation='Additivity is mathematical completeness; repeatability and small-perturbation stability are not measures of clinical or technician usefulness.'))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
