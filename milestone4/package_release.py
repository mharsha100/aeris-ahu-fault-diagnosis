"""Package reviewable runnable implementation without raw archives or environments."""
from pathlib import Path
import hashlib,json,zipfile,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'dist/downloads';out.mkdir(parents=True,exist_ok=True)
def package(name,files):
 path=out/name
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(set(files)):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
 print(name,path.stat().st_size,flush=True)
 return {'name':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
code=[]
for folder in ['src','scripts','milestone3','milestone4','docs','experiments']:
 code.extend((ROOT/folder).rglob('*'))
code.extend((ROOT/'dist'/n for n in ['app.js','index.html','inference.js','style.css']))
code.extend(ROOT/n for n in ['package.json','app.py','README.md','AGENTS.md','requirements.txt','requirements-cuda.txt','requirements-training.txt'])
records=[package('aeris_m4_code_and_evidence.zip',code)]
# Ready-to-run packs include only the fixed inputs and metadata needed for app/training.
files=list((ROOT/'data/aeris_windows_v1').glob('*'))+list((ROOT/'data/aeris_windows_v1/evaluation').glob('*'))
records.append(package('aeris_m4_prepared_inputs.zip',files))
# Store identical fitted forests once; restore exact bytes before loading bundles.
blobs=ROOT/'data/aeris_model_blobs';blobs.mkdir(parents=True,exist_ok=True)
installation={};metadata=[];blob_names={}
for n in ['m4_v0','m4_v1','m4_v2']:
 for suffix in ['', '_cross_adapted','_feedback']:
  bundle=ROOT/'data/aeris_models'/f'{n}{suffix}'
  config=json.loads((bundle/'hybrid.json').read_text());sha=config['artifact_sha256']['forest.joblib'];dest=blobs/f'{sha}.joblib'
  if not dest.exists():shutil.copyfile(bundle/'forest.joblib',dest)
  installation[bundle.name]=sha;blob_names.setdefault(sha,[]).append(bundle.name)
  metadata.extend(p for p in bundle.rglob('*') if p.name!='forest.joblib')
install=ROOT/'data/model_install.json';install.write_text(json.dumps(installation,indent=2));metadata.append(install)
records.append(package('aeris_m4_model_metadata.zip',metadata))
forest_artifacts=[]
for i,(sha,names) in enumerate(blob_names.items()):
 name=f'aeris_m4_forest_{i+1}.zip';records.append(package(name,[blobs/f'{sha}.joblib']))
 forest_artifacts.append({'label':'Shared initial RF weights (V0/V1/V2)' if len(names)>3 else 'Feedback RF weights · '+names[0].replace('m4_','').replace('_feedback','').upper(),'url':'downloads/'+name})
records.append(package('aeris_m4_evidence.zip',list((ROOT/'milestone4/evidence').rglob('*'))))
subprocess.run(['git','bundle','create',str(out/'aeris_m4_history.bundle'),'HEAD'],cwd=ROOT,check=True)
(ROOT/'dist/release_manifest.json').write_text(json.dumps({'artifacts':records},indent=2))
m=json.loads((ROOT/'dist/manifest.json').read_text());m['artifacts']=[{'label':'Runnable code + evidence','url':'downloads/aeris_m4_code_and_evidence.zip'},{'label':'Prepared input packs','url':'downloads/aeris_m4_prepared_inputs.zip'},{'label':'All model configurations + temporal weights','url':'downloads/aeris_m4_model_metadata.zip'}]+forest_artifacts+[{'label':'Experiment evidence','url':'downloads/aeris_m4_evidence.zip'},{'label':'Implementation Git history','url':'downloads/aeris_m4_history.bundle'},{'label':'Verification report','url':'validation_report.json'}];(ROOT/'dist/manifest.json').write_text(json.dumps(m,indent=2))
