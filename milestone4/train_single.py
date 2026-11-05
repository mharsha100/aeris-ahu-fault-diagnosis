"""Execute one frozen architecture configuration, for bounded CPU concurrency."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_experiment import verify_readiness
from aeris.training import train_source_hybrid
if __name__=='__main__':
 version=sys.argv[1]
 if version not in ['v0','v1','v2']:raise ValueError('Architecture required')
 train_source_hybrid(ROOT,model_version=version,run_name='m4_'+version,device='cpu',epochs=50,patience=0,seed=42,batch_size=256,trees=200,jobs=-1,fusion_weight=.5,experiment_scope=verify_readiness(ROOT))
