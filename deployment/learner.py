from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.learning import worker
from aeris.operations import Ledger
worker(ROOT,Ledger(os.environ.get('AERIS_DB','/state/ledger.sqlite3')))
