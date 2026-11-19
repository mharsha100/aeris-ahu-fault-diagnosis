"""Repair the booster continuation while retaining an audited identical partial fit."""
from pathlib import Path
import sys,json,uuid,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.learning import Learning,train_candidate
from aeris.operations import Ledger
l=Learning(ROOT,Ledger(os.environ.get('AERIS_DB','/state/ledger.sqlite3')))
with l.ledger.db() as c:r=c.execute("SELECT * FROM learning_jobs WHERE cohort='test' AND stage='v1' AND status='rejected' ORDER BY created DESC LIMIT 1").fetchone()
if not r:raise ValueError('No rejected V1 candidate to repair')
old=json.loads((l.home/'models'/r['bundle']/'training_manifest.json').read_text());jid=uuid.uuid4().hex
job={**old,'id':jid,'bundle':'adaptive_test_v1_'+jid[:12],'reuse_partial_fit':r['bundle']}
with l.ledger.db() as c:
 c.execute('INSERT INTO learning_jobs VALUES(?,?,?,?,?,?,?,?,?,?)',(jid,'test','v1',job['parent'],'training',time.time(),None,job['bundle'],None,None))
 for s in job['samples']:c.execute("UPDATE learning_samples SET state='training' WHERE cohort='test' AND digest=?",(s['digest'],))
try:print('V1 REPAIR',l.finish(job,train_candidate(l,job)),flush=True)
except Exception as e:l.finish(job,error=str(e));raise
