from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
while not all((ROOT/'data/aeris_models'/name/'READY.json').exists() for name in ['m4_v0','m4_v1','m4_v2']):time.sleep(5)
if not (ROOT/'milestone4/evidence/architecture_training.json').exists():
 subprocess.run([sys.executable,'milestone4/train_architectures.py','--device','cpu','--resume'],cwd=ROOT,check=True)
