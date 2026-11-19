"""Explicit container/host commands; never trains during normal demo startup."""
from pathlib import Path
import argparse,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('train');t.add_argument('--version',choices=['v0','v1','v2'],required=True);t.add_argument('--run-name',required=True);t.add_argument('--epochs',type=int,default=50);t.add_argument('--seed',type=int,default=42);t.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    s.add_parser('preprocess');s.add_parser('evaluate');args=p.parse_args()
    if args.command=='train':
        from aeris.training import train_source_hybrid
        train_source_hybrid(ROOT,model_version=args.version,run_name=args.run_name,epochs=args.epochs,patience=0,device=args.device,seed=args.seed,fusion_weight=.5,jobs=2)
    elif args.command=='evaluate':subprocess.run([sys.executable,str(ROOT/'scripts/10_evaluate.py')],check=True)
    else:
        if (ROOT/'data/aeris_windows_v1/READY.json').exists():raise SystemExit('Frozen prepared data already exist; preserving them. For a rebuild use a separate data mount containing raw inputs and no prepared artifacts.')
        for name in ['01_inventory.py','02_prepare_data.py','03_make_splits_windows.py']:subprocess.run([sys.executable,str(ROOT/'scripts'/name)],check=True)
if __name__=='__main__':main()
