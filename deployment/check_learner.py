import sqlite3,time,os
c=sqlite3.connect(os.environ.get('AERIS_DB','/state/ledger.sqlite3'))
r=c.execute("SELECT value FROM learning_control WHERE key='heartbeat'").fetchone()
raise SystemExit(0 if r and time.time()-float(r[0])<90 else 1)
