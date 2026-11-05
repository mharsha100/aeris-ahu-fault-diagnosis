"""Run each architecture through target gating and simulated verified feedback."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def one(name):
 from aeris.adaptation import adapt_hybrid
 from aeris.workflow import feedback_update
 if not (ROOT/'data/aeris_models'/f'{name}_cross_adapted/READY.json').exists():adapt_hybrid(ROOT,model_name=name,device='cpu',epochs=5)
 if not (ROOT/'data/aeris_models'/f'{name}_feedback/feedback_report.json').exists():feedback_update(ROOT,name)
if __name__=='__main__':
 if len(sys.argv)>1:one(sys.argv[1])
 else:
  jobs=[]
  for name in ['m4_v0','m4_v1','m4_v2']:
   log=(ROOT/'data'/f'{name}_learning.log').open('w')
   jobs.append((name,subprocess.Popen([sys.executable,__file__,name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log))
  for name,process,log in jobs:
   code=process.wait();log.close()
   if code:raise RuntimeError(f'{name} learning failed; see data/{name}_learning.log')
   print(name+' adaptation/feedback complete',flush=True)
  print('All architecture-specific adaptation and feedback workflows finished.',flush=True)
