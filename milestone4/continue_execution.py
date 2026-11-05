"""Continue the actual training workflow; never substitutes fixtures for results."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
print('Waiting for three complete50-epoch bundles…',flush=True)
while not (ROOT/'milestone4/evidence/architecture_training.json').exists():
 time.sleep(5)
for file in ['run_learning.py','evaluate_pipeline.py','check_pipeline.py','export_browser.py']:
 print('Running',file,flush=True)
 subprocess.run([sys.executable,str(ROOT/'milestone4'/file)],cwd=ROOT,check=True)
subprocess.run(['node','milestone4/browser_parity.mjs'],cwd=ROOT,check=True)
print('Backend and numerical browser verification complete.',flush=True)
