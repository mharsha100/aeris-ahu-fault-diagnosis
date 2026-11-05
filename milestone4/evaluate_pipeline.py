"""Freeze source selection, then report retrospective pipeline comparisons."""
from pathlib import Path
import json,sys,time,platform,resource,subprocess
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.modeling import load_hybrid,metrics,CLASS_NAMES,sha256_file
from aeris.workflow import diagnose,canonical_window
from aeris.training import write_json

def run():
 torch.set_num_threads(2);w=ROOT/'data/aeris_windows_v1';out=ROOT/'milestone4/evidence';out.mkdir(parents=True,exist_ok=True)
 val={k:v for k,v in np.load(w/'source_validation.npz',allow_pickle=False).items()}
 meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id');vmeta=meta.loc[val['window_id']].reset_index()
 rows=[];details={};examples=[];thresholds=[];dayrows=[];candidates=[];timings=[];investigations=[];paired={}
 for name in ['m4_v0','m4_v1','m4_v2']:
  b=ROOT/'data/aeris_models'/name;m=load_hybrid(b);m.forest.n_jobs=1;p=m.predict(val['X_seq'],val['X_tab']);score=metrics(val['y'],p['probabilities'])
  missed=int(((val['y']!=0)&(p['class_id']==0)).sum());candidates.append({'name':name,'macro_f1':score['macro_f1'],'fault_predicted_normal':missed,'parameters':sum(x.numel() for x in m.temporal.parameters())})
  for threshold in [.80,.90,.95]:
   review=(p['confidence']<threshold)|p['branch_disagreement'];wrong=p['class_id']!=val['y']
   thresholds.append({'model':name,'threshold':threshold,'review_fraction':float(review.mean()),'wrong_not_reviewed':int((wrong&~review).sum()),'confidently_wrong':int((wrong&(p['confidence']>=threshold)).sum()),'windows':len(wrong)})
 best=max(c['macro_f1'] for c in candidates);eligible=[c for c in candidates if c['macro_f1']>=best-.005]
 winner=min(eligible,key=lambda c:(c['fault_predicted_normal'],c['parameters']))
 selection={'selected_initial_bundle':winner['name'],'decision':'Among models within .005 of best source-validation macro F1, prefer fewer fault→normal errors, then fewer parameters.',
  'candidates':candidates,'basis':'Source validation only; no target benchmark used to select the initial component',
  'provisional_research_selection':True,'architecture_constraint':'User-confirmed hybrid research design. RF-only is included as a reference; no claim that adding temporal modeling is beneficial when scores do not improve.','target_safety_or_transfer_validated':False,'risk':'False normal predictions and unsupported transfer remain consequential. All target requests require review.',
  'latency_requirement':'95/100 full warm UI requests under5s; measured separately','paid_model_api_cost_usd':0,'hardware_cost':'Not metered; zero API charge does not imply free compute.',
  'complexity':'All share RF+TCN. Attention and combined summaries add parameters, versioned calibration, and deployment code.',
  'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 write_json(out/'component_selection.json',selection);pd.DataFrame(thresholds).to_csv(out/'review_threshold_experiment.csv',index=False)
 # Selection is frozen BEFORE opening retrospective benchmark labels.
 target={k:v for k,v in np.load(w/'target_test.npz',allow_pickle=False).items()};source_test={k:v for k,v in np.load(w/'source_test.npz',allow_pickle=False).items()}
 ref=pd.read_csv(w/'evaluation/target_window_reference.csv.gz').set_index('window_id');target['y']=ref.loc[target['window_id'],'class_id'].to_numpy('int64')
 packs={'source_validation':val,'source_test_within_condition':source_test,'target_retrospective':target}
 for name in ['m4_v0','m4_v1','m4_v2']:
  for suffix in ['', '_cross_adapted','_feedback']:
   bundle=name+suffix;b=ROOT/'data/aeris_models'/bundle;m=load_hybrid(b);m.forest.n_jobs=1
   for role,pack in packs.items():
    p=m.predict(pack['X_seq'],pack['X_tab']);score=metrics(pack['y'],p['probabilities']);details[f'{bundle}/{role}']=score
    if suffix=='':paired[(name,role)]={'correct':p['class_id']==pack['y'],'prediction':p['class_id']}
    row={'model':bundle,'partition':role,'windows':len(pack['y']),'macro_f1':score['macro_f1'],'accuracy':score['accuracy'],
     'normal_recall':score['per_class']['normal']['recall'],'fault_to_normal':int(((pack['y']!=0)&(p['class_id']==0)).sum()),
     'model_review_fraction':float(p['needs_review'].mean()),'confident_wrong_090':int(((pack['y']!=p['class_id'])&(p['confidence']>=.9)).sum()),
     'setting_review_required':role=='target_retrospective','independent_condition_claim_allowed':False}
    rows.append(row)
    mm=meta.loc[pack['window_id']].reset_index();mm['truth']=pack['y'];mm['predicted']=p['class_id'];mm['confidence']=p['confidence'];mm['model']=bundle;mm['role']=role
    mm['forest_prediction']=[CLASS_NAMES[k] for k in p['forest_probabilities'].argmax(axis=1)]
    mm['temporal_prediction']=[CLASS_NAMES[k] for k in p['temporal_probabilities'].argmax(axis=1)]
    mm['branch_disagreement']=p['branch_disagreement'];mm['model_review_required']=p['needs_review']
    mm['max_source_z']=np.abs(pack['X_seq']).max(axis=(1,2))
    for feature in ['oa_damper_cmd','cooling_valve_cmd','heating_valve_cmd','supply_minus_mixed_c']:
     mm[feature+'_mean']=pack['X_tab'][:,m.protocol['tabular_features'].index(feature+'__mean')]
    if suffix=='':
     misses=mm.loc[(mm.truth!=0)&mm.predicted.eq(0)]
     reference_normal=mm.loc[mm.truth.eq(0)]
     investigations.append({'model':bundle,'partition':role,'pattern':'Fault predicted normal','affected_windows':len(misses),
      'high_confidence_misses':int(misses.confidence.ge(.9).sum()),'unreviewed_model_misses':int((~misses.model_review_required).sum()),
      'median_max_source_z_misses':float(misses.max_source_z.median()) if len(misses) else None,
      'median_oa_command_misses':float(misses.oa_damper_cmd_mean.median()) if len(misses) else None,
      'median_cooling_command_misses':float(misses.cooling_valve_cmd_mean.median()) if len(misses) else None,
      'median_heating_command_misses':float(misses.heating_valve_cmd_mean.median()) if len(misses) else None,
      'median_supply_minus_mixed_c_misses':float(misses.supply_minus_mixed_c_mean.median()) if len(misses) else None,
      'median_supply_minus_mixed_c_reference_normal':float(reference_normal.supply_minus_mixed_c_mean.median()) if len(reference_normal) else None,
      'branch_disagreement_misses':int(misses.branch_disagreement.sum()),
      'action':'Require technician review on every exploratory target. Do not turn confident normal into an automatic no-fault clearance.',
      'interpretation':'Observed command/temperature associations are diagnostic clues, not causal explanations. Scenario condition labels do not establish observable faults at every minute.'})
    for group,g in mm.groupby('condition_group'):
     dayrows.append({'model':bundle,'partition':role,'condition_group':group,'windows':len(g),'accuracy':float((g.truth==g.predicted).mean()),'independent_weather_conditions':False})
    if suffix=='':
     wrong=mm.loc[mm.truth!=mm.predicted].sort_values('confidence',ascending=False)
     wrong['category']=np.where((wrong.truth!=0)&wrong.predicted.eq(0),'Fault predicted normal',np.where(wrong.truth.eq(0),'False fault alarm','Wrong fault family'))
     cases=wrong.drop_duplicates(['truth','predicted','run_id','condition_group']).head(20)
     for r in cases.to_dict('records'):
      r['true_class']=CLASS_NAMES[r['truth']];r['predicted_class']=CLASS_NAMES[r['predicted']];r['confidently_wrong']=r['confidence']>=.9
      r['is_independent_case']=False;examples.append(r)
    if suffix=='' and role=='source_validation':
     # RF-only and TCN-only ablations share the same fitted branches/calibration.
     for branch in ['forest','temporal']:
      bs=metrics(pack['y'],p[f'{branch}_probabilities']);rows.append({'model':name+'_'+branch+'_only','partition':role,'windows':len(pack['y']),'macro_f1':bs['macro_f1'],'accuracy':bs['accuracy'],'normal_recall':bs['per_class']['normal']['recall'],'independent_condition_claim_allowed':False})
   rng=np.random.default_rng(42);ixs=rng.choice(len(val['y']),100,replace=False)
   for ix in ixs[:3]:diagnose(m,canonical_window(val,ix,m.scaler),setting=vmeta.iloc[ix].season,timestamps=np.arange(15)*60,recorded_source=True)
   for ix in ixs:
    start=time.perf_counter();diagnose(m,canonical_window(val,ix,m.scaler),setting=vmeta.iloc[ix].season,timestamps=np.arange(15)*60,recorded_source=True);timings.append({'model':bundle,'window_id':str(val['window_id'][ix]),'seconds':time.perf_counter()-start})
   print(f'{bundle}: evaluation and100 full decision requests complete',flush=True)
 pairs=[]
 for role in packs:
  for left,right in [('m4_v0','m4_v1'),('m4_v1','m4_v2')]:
   a,b=paired[(left,role)],paired[(right,role)]
   pairs.append({'partition':role,'left':left,'right':right,'rescued_wrong_windows':int((~a['correct']&b['correct']).sum()),'new_wrong_windows':int((a['correct']&~b['correct']).sum()),'same_window_descriptive_only':True})
 pd.DataFrame(pairs).to_csv(out/'paired_window_changes.csv',index=False)
 write_json(out/'consequential_error_investigation.json',{'pattern':'Fault predicted normal','analyses':investigations,'independent_case_inference':False})
 pd.DataFrame(rows).to_csv(out/'results.csv',index=False);pd.DataFrame(examples).to_csv(out/'error_examples.csv',index=False);pd.DataFrame(dayrows).to_csv(out/'by_day.csv',index=False)
 tf=pd.DataFrame(timings);tf.to_csv(out/'request_timing_samples.csv',index=False)
 summaries=[]
 for name,g in tf.groupby('model'):summaries.append({'model':name,'requests':len(g),'p50_ms':float(g.seconds.median()*1000),'p95_ms':float(g.seconds.quantile(.95)*1000),'under_5s':int(g.seconds.lt(5).sum()),'included':'Canonical window reconstruction, validation, feature extraction, RF/TCN, calibration, scope/review decision','excluded':'Model load, network and browser render','device':'CPU','torch_threads':2,'forest_jobs':1})
 pd.DataFrame(summaries).to_csv(out/'timing_summary.csv',index=False);write_json(out/'metric_details.json',details)
 # Export inspectable error categories and a consequential pattern investigation.
 ef=pd.DataFrame(examples)
 # A bounded20-case review set spans models, pools and consequential patterns.
 groups=[g.sort_values('confidence',ascending=False).to_dict('records') for _,g in ef.groupby(['model','role','category'],sort=True)]
 review=[]
 while len(review)<20 and any(groups):
  for group in groups:
   if group and len(review)<20:review.append(group.pop(0))
 pd.DataFrame(review).to_csv(out/'error_review_20.csv',index=False)
 ef.groupby(['model','role','category']).size().reset_index(name='sampled_windows').to_csv(out/'error_categories.csv',index=False)
 write_json(out/'evaluation_protocol.json',{'source_validation_used_for_selection':True,'source_test_status':'Later days in same3conditions, descriptive only',
 'target_status':'Retrospective exploratory: published labels exposed in earlier work; no sealed holdout claim',
 'test_roles_used_for_training':False,'prediction_unit':'15minwindow stride5','condition_unit':'3seasonalsharedweatherweeks',
 'error_sampling':'Up to20 highest confidence errors per initial model/pool, diversified by true/predicted/run/day; not random prevalence sample',
 'consequential_pattern':'Count actual fault→normal errors in results.csv and inspect their confidence, setting and branch outputs; event labels do not establish active physical fault at every minute.',
 'confidence_threshold_status':'Provisional; the threshold experiment measures tradeoffs on development recordings only.',
 'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'python':platform.python_version(),
 'source_pack_sha256':{role:sha256_file(w/f'{role}.npz') for role in ['source_train','source_validation','source_calibration','source_test','target_adapt','target_feedback','target_test']}})
 print('Component selected:',winner['name'],flush=True)
if __name__=='__main__':run()
