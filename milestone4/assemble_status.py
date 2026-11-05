"""Collect the completion gates; called after real browser verification."""
from pathlib import Path
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];ev=ROOT/'milestone4/evidence'
def read(p):return json.loads(p.read_text())
checks=read(ev/'validation_report.json');parity=read(ev/'browser_numeric_parity.json');browser=read(ev/'browser_ui_verification.json')
if checks['status']!='passed' or parity['status']!='passed' or browser['status']!='passed':raise RuntimeError('Incomplete verification')
status={'version':'m4-implementation-v1','implementation_status':'complete_for_research_milestone','documentation_started':False,
 'initial_architectures':'Fresh RF+TCN V0/V1/V2,50completedepochs each; best checkpoint selection disclosed',
 'models':{},'automated_check_count':checks['passed_checks'],'browser_parity':parity['status'],'browser_UI':browser['status'],
 'professor_M3_feedback':{'feasible_question':'Descriptive within-recording comparison; original five-independent-case floor retained',
 'channel_mapping':'47rawfiles +50conversion checks; units,roles,modes,command limitations and supported settings recorded',
 'reproduction':'Preserved m3-v1; actual fresh Windows target data check passed, fresh Linux dependencies and full uncached source download passed; blocked cloud target attempt retained'},
 'claims_not_established':['Independent class/season generalization','Transfer to unseen buildings','Target calibration safety','Real technician intervention','Automatically safe equipment action'],
 'publication':{'original_GitHub_push':'blocked: integration write403 and CLI has no GitHub credentials','local_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'ready_for_documentation':True,'instructor_access_to_new_GitHub_branch':'Requires repository write access before submission; implementation/source history is packaged separately'},
 'selected_initial_component':read(ev/'component_selection.json')['selected_initial_bundle']}
for n in ['m4_v0','m4_v1','m4_v2']:
 b=ROOT/'data/aeris_models';status['models'][n]={'initial_epochs':read(b/n/'training_run.json')['completed_temporal_epochs'],
 'adaptation_status':read(b/f'{n}_cross_adapted/hybrid.json')['adaptation_status'],'feedback_status':read(b/f'{n}_feedback/feedback_report.json')['status']}
(ev/'pipeline_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
