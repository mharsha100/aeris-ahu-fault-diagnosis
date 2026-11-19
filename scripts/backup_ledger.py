"""Online SQLite backup plus a separate, non-destructive restore rehearsal."""
import argparse,sqlite3,json,hashlib,tempfile,os
from pathlib import Path

def signature(connection):
    out={}
    for table,key in [('predictions','id'),('feedback','prediction_id'),('feedback_history','id')]:
        rows=connection.execute(f'SELECT * FROM {table} ORDER BY {key}').fetchall()
        out[table]={'rows':len(rows),'sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()}
    return out

def backup(source,destination):
    source=Path(source);destination=Path(destination)
    if not source.is_file():raise ValueError('Ledger does not exist')
    if destination.exists():raise ValueError('Choose a new backup filename; existing backups are preserved')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as live,sqlite3.connect(destination) as snapshot:
        live.backup(snapshot)
        assert snapshot.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert not snapshot.execute('PRAGMA foreign_key_check').fetchall()
        expected=signature(snapshot)
        with tempfile.TemporaryDirectory(prefix='aeris-restore-') as tmp:
            with sqlite3.connect(str(Path(tmp)/'restored.sqlite3')) as restored:
                snapshot.backup(restored)
                assert restored.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                assert signature(restored)==expected
    return dict(backup=str(destination),tables=expected,integrity='ok',foreign_keys='ok',restored_rows_and_content_identical=True,production_ledger_modified=False,scope='Online snapshot restored into a disposable independent database. Not a disaster-recovery time objective or off-host storage test.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=os.environ.get('AERIS_DB','/state/ledger.sqlite3'));p.add_argument('--destination',required=True);a=p.parse_args();print(json.dumps(backup(Path(a.source).resolve(),Path(a.destination).resolve()),indent=2))
