"""Durable, cohort-isolated V0 -> pseudo-label V1 -> reviewed-label V2 lifecycle.

The worker trains outside the inference process. SQLite transactions publish an
immutable, hash-checked bundle only after source-retention checks. Test records
can update only the test model. Evaluation recordings never enter the queue.
"""
from pathlib import Path
import copy, hashlib, json, os, time, uuid
import numpy as np
from .operations import utc

MIN_BATCH = 8
STAGES = {'v0': 'source_trained', 'v1': 'confident_pseudo_labels', 'v2': 'technician_feedback'}

def fingerprint(raw):
    # Stable across JSON/CSV float round trips. Also catches exported holdouts.
    return hashlib.sha256(np.round(np.asarray(raw, dtype='float64'), 3).tobytes()).hexdigest()

class Learning:
    def __init__(self, root, ledger):
        self.root, self.ledger = Path(root), ledger
        self.home = ledger.path.parent/'adaptation'; self.home.mkdir(parents=True, exist_ok=True)
        self.protected = None
        with ledger.db() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS learning_samples(
              cohort TEXT, digest TEXT, prediction_id TEXT REFERENCES predictions(id),
              kind TEXT, label INTEGER, confidence REAL, raw TEXT, provenance TEXT,
              revision INTEGER NOT NULL DEFAULT 1, state TEXT, updated REAL,
              PRIMARY KEY(cohort,digest));
            CREATE TABLE IF NOT EXISTS learning_jobs(
              id TEXT PRIMARY KEY, cohort TEXT, stage TEXT, parent TEXT, status TEXT,
              created REAL, finished REAL, bundle TEXT, report TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS learning_registry(
              cohort TEXT PRIMARY KEY, stage TEXT, bundle TEXT, job_id TEXT, updated REAL);
            CREATE TABLE IF NOT EXISTS learning_control(key TEXT PRIMARY KEY, value TEXT);
            ''')

    def source(self):
        selection=self.root/'data/deployment_selection.json'
        return json.loads(selection.read_text(encoding='utf-8-sig'))['bundle'] if selection.exists() else os.environ.get('AERIS_BUNDLE','v1_attention')

    def active(self, cohort):
        if cohort not in ('test','genuine'): raise ValueError('Invalid cohort')
        with self.ledger.db() as c: row=c.execute('SELECT * FROM learning_registry WHERE cohort=?',(cohort,)).fetchone()
        return dict(row) if row else dict(cohort=cohort,stage='v0',bundle=self.source(),job_id=None)

    def bundle_path(self, name):
        if not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name: raise ValueError('Invalid model name')
        with self.ledger.db() as c:
            row=c.execute("SELECT 1 FROM learning_jobs WHERE bundle=? AND status='promoted'",(name,)).fetchone()
        return self.home/'models'/name if row else self.root/'data/aeris_models'/name

    def is_protected(self, raw, scaler):
        if self.protected is None:
            self.protected=set()
            for role in ['source_train','source_validation','source_calibration','source_test','target_test']:
                with np.load(self.root/f'data/aeris_windows_v1/{role}.npz') as p:
                    for seq in p['X_seq']:
                        x=(seq.astype('float64')*np.asarray(scaler['scale'])+np.asarray(scaler['mean'])).astype('float32')
                        self.protected.add(fingerprint(x))
        return fingerprint(raw) in self.protected

    def context(self, body, recorded, raw, model):
        if recorded:
            return body.get('role') in ('target_adapt','target_feedback')
        return (body.get('context_verified') is True and body.get('setting') in ('SZVAV','SZCAV')
                and not self.is_protected(raw, model.scaler))

    def observe(self, record, raw, eligible, provenance):
        result=record['result']
        if 'class_id' not in result: return dict(route='blocked',reason='Invalid input cannot train.')
        if not eligible: return dict(route='evaluation_only',reason='Evaluation or unverified-context input; excluded from training.')
        if result['status']=='review_required' or result['confidence']<.95:
            return dict(route='awaiting_technician',reason='A reviewed class, reviewer identity and evidence are needed for V2.')
        return self.enqueue(record,raw,'pseudo',result['class_id'],result['confidence'],provenance)

    def enqueue(self, record, raw, kind, label, confidence, provenance):
        digest=fingerprint(raw); cohort=record['cohort']
        with self.ledger.db() as c:
            old=c.execute('SELECT * FROM learning_samples WHERE cohort=? AND digest=?',(cohort,digest)).fetchone()
            # Never overwrite a technician label with the model's own guess.
            if old and (old['kind']=='review' and kind=='pseudo' or old['kind']==kind and old['label']==label):
                return dict(route='already_recorded',kind=old['kind'],state=old['state'],reason='This sensor window is already in the learning ledger.')
            count=c.execute('SELECT COUNT(*) FROM learning_samples WHERE cohort=?',(cohort,)).fetchone()[0]
            if not old and count>=256: return dict(route='capacity_reached',reason='256-window research adaptation budget reached; inspect the accumulated evidence before expanding it.')
            c.execute('''INSERT INTO learning_samples VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(cohort,digest) DO UPDATE SET prediction_id=excluded.prediction_id,kind=excluded.kind,
            label=excluded.label,confidence=excluded.confidence,provenance=excluded.provenance,
            revision=learning_samples.revision+1,state='pending',updated=excluded.updated''',
            (cohort,digest,record['prediction_id'],kind,int(label),float(confidence),json.dumps(raw),provenance,1,'pending',time.time()))
        return dict(route='queued_v1' if kind=='pseudo' else 'queued_v2',kind=kind,reason='Saved for automatic background training; a candidate must pass evaluation before activation.')

    def feedback(self, pid, body):
        with self.ledger.db() as c:
            queued=c.execute("SELECT 1 FROM learning_samples WHERE prediction_id=? AND kind='review'",(pid,)).fetchone()
        if queued and not (body.get('verified_class') or body.get('corrected_class')):
            raise ValueError('This label is already in the learning history. Submit a corrected class and evidence; an empty review cannot silently revoke a trained label.')
        with self.ledger.db() as c: row=c.execute('SELECT * FROM predictions WHERE id=?',(pid,)).fetchone()
        if row is None: raise ValueError('Unknown prediction ID')
        record=self.ledger.decode(row); r=record['result']
        label=body.get('verified_class') or (body.get('corrected_class') if record['cohort']=='test' else None)
        eligible=bool(label and r.get('adaptation_eligible') and str(body.get('inspector') or '').strip() and str(body.get('provenance') or '').strip())
        if queued and not eligible:
            raise ValueError('An existing learning label requires its confirmed class, reviewer and evidence when editing the review.')
        clean=self.ledger.feedback(pid,body,training_eligible=eligible)
        if eligible:
            from .modeling import CLASS_NAMES
            clean['learning']=self.enqueue(record,json.loads(row['raw']),'review',CLASS_NAMES.index(label),1.,clean['provenance'])
            clean['training_eligible']=True
        else:
            clean['learning']=dict(route='review_saved_only',reason='Training requires an eligible adaptation input, a confirmed class, reviewer identity and evidence. Genuine use requires the independently inspected label.')
        return clean

    def status(self):
        with self.ledger.db() as c:
            counts=[dict(r) for r in c.execute('SELECT cohort,kind,state,COUNT(*) AS count FROM learning_samples GROUP BY cohort,kind,state')]
            jobs=[dict(r) for r in c.execute('SELECT * FROM learning_jobs ORDER BY created DESC LIMIT 20')]
            heartbeat=c.execute("SELECT value FROM learning_control WHERE key='heartbeat'").fetchone()
        for j in jobs: j['report']=json.loads(j['report']) if j['report'] else None
        return dict(active={c:self.active(c) for c in ('test','genuine')},samples=counts,jobs=jobs,minimum_batch=MIN_BATCH,
                    worker_alive=bool(heartbeat and time.time()-float(heartbeat[0])<90),
                    explanation='V0 learns from source data. V1 adds confident, unreviewed labels. V2 adds reviewed labels. Training runs in batches; promotion is conditional, not a promise of improved target accuracy.')

    def metrics(self):
        s=self.status(); lines=[f'aeris_learning_worker_up {int(s["worker_alive"])}']
        for cohort,a in s['active'].items():
            lines.append(f'aeris_learning_stage{{cohort="{cohort}"}} {int(a["stage"][1:])}')
            for kind in ('pseudo','review'):
                for state in ('pending','training','used','rejected'):
                    n=sum(x['count'] for x in s['samples'] if x['cohort']==cohort and x['kind']==kind and x['state']==state)
                    lines.append(f'aeris_learning_samples{{cohort="{cohort}",kind="{kind}",state="{state}"}} {n}')
            with self.ledger.db() as c:
                for state in ('training','promoted','rejected','failed'):
                    n=c.execute('SELECT COUNT(*) FROM learning_jobs WHERE cohort=? AND status=?',(cohort,state)).fetchone()[0]
                    lines.append(f'aeris_learning_jobs{{cohort="{cohort}",status="{state}"}} {n}')
        return '\n'.join(lines)+'\n'

    def claim(self):
        with self.ledger.db() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute("SELECT 1 FROM learning_jobs WHERE status='training'").fetchone(): return None
            for cohort in ('genuine','test'):
                pending=c.execute("SELECT * FROM learning_samples WHERE cohort=? AND state='pending' ORDER BY updated",(cohort,)).fetchall()
                if len(pending)<MIN_BATCH or time.time()-pending[-1]['updated']<15: continue
                rows=c.execute("SELECT * FROM learning_samples WHERE cohort=? ORDER BY updated",(cohort,)).fetchall()
                stage='v2' if any(r['kind']=='review' for r in rows) else 'v1'
                parent=self.active(cohort)['bundle']; jid=uuid.uuid4().hex; name=f'adaptive_{cohort}_{stage}_{jid[:12]}'
                c.execute('INSERT INTO learning_jobs VALUES(?,?,?,?,?,?,?,?,?,?)',(jid,cohort,stage,parent,'training',time.time(),None,name,None,None))
                c.execute("UPDATE learning_samples SET state='training' WHERE cohort=? AND state='pending'",(cohort,))
                return dict(id=jid,cohort=cohort,stage=stage,parent=parent,bundle=name,samples=[dict(r) for r in rows])
        return None

    def finish(self, job, report=None, error=None):
        with self.ledger.db() as c:
            c.execute('BEGIN IMMEDIATE')
            changed=any(not c.execute('SELECT 1 FROM learning_samples WHERE cohort=? AND digest=? AND revision=?',(job['cohort'],r['digest'],r['revision'])).fetchone() for r in job['samples'])
            active=c.execute('SELECT bundle FROM learning_registry WHERE cohort=?',(job['cohort'],)).fetchone()
            if (active['bundle'] if active else self.source())!=job['parent']:
                changed=True
            if changed: error='Training labels changed during this run; candidate withheld and fresh training queued.'
            accepted=bool(report and report['guard']['passed'] and not error)
            status='promoted' if accepted else ('failed' if error else 'rejected')
            c.execute('UPDATE learning_jobs SET status=?,finished=?,report=?,error=? WHERE id=?',(status,time.time(),json.dumps(report) if report else None,error,job['id']))
            for r in job['samples']:
                c.execute('UPDATE learning_samples SET state=? WHERE cohort=? AND digest=? AND revision=? AND state=\'training\'',('pending' if changed else 'used' if accepted else 'rejected',job['cohort'],r['digest'],r['revision']))
            if accepted:
                c.execute('INSERT INTO learning_registry VALUES(?,?,?,?,?) ON CONFLICT(cohort) DO UPDATE SET stage=excluded.stage,bundle=excluded.bundle,job_id=excluded.job_id,updated=excluded.updated',(job['cohort'],job['stage'],job['bundle'],job['id'],time.time()))
        return status

def train_candidate(learning,job):
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from catboost import CatBoostClassifier
    from .boosted import BoostedForest
    from .physical import PhysicalFeatures
    from .features import model_inputs
    from .modeling import load_hybrid, HybridPredictor, metrics, fit_temperature, temperature_probabilities
    from .training import load_source_partition, write_json
    from .adaptation import fine_tune_temporal, recalibrate, source_guardrail
    from .workflow import write_bundle
    torch.set_num_threads(2); torch.manual_seed(57); np.random.seed(57)
    parent=learning.bundle_path(job['parent']); m=load_hybrid(parent); m.forest.n_jobs=2
    w=learning.root/'data/aeris_windows_v1'; parts={p:load_source_partition(w,'source_'+p,m.protocol) for p in ('train','validation','calibration')}
    tr,val,cal=(parts[p] for p in ('train','validation','calibration'))
    raw=np.array([json.loads(s['raw']) for s in job['samples']],dtype='float32'); seq,tab=model_inputs(raw,m.scaler,m.protocol)
    y=np.array([s['label'] for s in job['samples']],dtype='int64'); confidence=np.array([s['confidence'] for s in job['samples']],dtype='float32')
    before=metrics(val['y'],m.predict(val['X_seq'],val['X_tab'])['probabilities'])
    base=m.forest.forest if isinstance(m.forest,BoostedForest) else m.forest
    forest=RandomForestClassifier(**base.get_params());forest.n_jobs=2
    weights=np.r_[np.ones(len(tr['y'])),[5. if s['kind']=='review' else 2.*s['confidence'] for s in job['samples']]]
    X=np.concatenate([tr['X_tab'],tab]); S=np.concatenate([tr['X_seq'],seq]); Y=np.r_[tr['y'],y]
    if job.get('reuse_partial_fit'):
        previous=learning.home/'models'/job['reuse_partial_fit']
        manifest=json.loads((previous/'training_manifest.json').read_text())
        keys=lambda samples:sorted((s['digest'],s['revision'],s['kind'],s['label']) for s in samples)
        if manifest['parent']!=job['parent'] or keys(manifest['samples'])!=keys(job['samples']):raise ValueError('Partial-fit recovery requires identical parent and training records')
        fitted=load_hybrid(previous);forest=fitted.forest.forest;forest.n_jobs=2;temporal=fitted.temporal
        prior_report=json.loads((previous/'learning_report.json').read_text());history=prior_report['training_history'];epoch=prior_report['retained_temporal_epoch']
    else:
        forest.fit(X,Y,sample_weight=weights)
        temporal=copy.deepcopy(m.temporal)
        history,epoch=fine_tune_temporal(temporal,tr,val,dict(X_seq=seq,X_tab=tab,y=y,confidence=confidence),device='cpu',epochs=50,batch_size=256,seed=57)
    config=copy.deepcopy(m.configuration)
    if config.get('boosting'):
        # Continue the fitted booster on source replay + adaptation records.
        params=m.forest.booster.get_params()
        params.pop('devices',None);params.update(iterations=120,learning_rate=.025,task_type='CPU',random_seed=57,thread_count=2,verbose=False,allow_writing_files=False,use_best_model=False)
        booster=CatBoostClassifier(**params)
        booster.fit(np.concatenate([PhysicalFeatures().transform(X),S.reshape(len(S),-1)],axis=1),Y,sample_weight=weights,init_model=m.forest.booster)
        forest=BoostedForest(forest,booster)
    candidate=HybridPredictor(forest,temporal,config,m.scaler,m.protocol,'cpu');recalibrate(candidate,cal)
    if config.get('boosting'):
        config['boosting']['temperature']=fit_temperature(forest.boosted_probabilities(cal['X_seq'],cal['X_tab']),cal['y'])
        config['boosting']['hybrid_temperature']=1.
        config['boosting']['hybrid_temperature']=fit_temperature(candidate.predict(cal['X_seq'],cal['X_tab'])['probabilities'],cal['y'])
    after=metrics(val['y'],candidate.predict(val['X_seq'],val['X_tab'])['probabilities'])
    source=load_hybrid(learning.root/'data/aeris_models'/learning.source());source.forest.n_jobs=2
    original=metrics(val['y'],source.predict(val['X_seq'],val['X_tab'])['probabilities'])
    g1,g0=source_guardrail(before,after),source_guardrail(original,after)
    guard=dict(passed=g1['passed'] and g0['passed'],parent=g1,v0=g0)
    output=learning.home/'models'/job['bundle'];output.parent.mkdir(parents=True,exist_ok=True)
    saved=write_bundle(output,candidate,parent,dict(model_version=job['stage'],learning_stage=STAGES[job['stage']],parent_bundle=job['parent'],source_architecture=config.get('temporal_kind'),target_data_used=True,test_data_used=False,adaptation_job=job['id'],cohort=job['cohort']))
    np.testing.assert_allclose(candidate.predict(seq,tab)['probabilities'],saved.predict(seq,tab)['probabilities'],atol=1e-6)
    report=dict(guard=guard,source_before=before,source_candidate=after,source_v0=original,
                samples=len(y),pseudo_labels=sum(s['kind']=='pseudo' for s in job['samples']),reviewed_labels=sum(s['kind']=='review' for s in job['samples']),
                completed_temporal_epochs=len(history),maximum_temporal_epochs=50,retained_temporal_epoch=epoch,additional_boosting_iterations=120,
                labels_are_simulated=job['cohort']=='test',training_history=history,
                limitation='Source retention is a promotion guard, not proof of target improvement. Demo labels and overlapping windows do not establish independent field accuracy.')
    report['reused_partial_fit']=job.get('reuse_partial_fit')
    report['new_temporal_epochs_this_attempt']=0 if job.get('reuse_partial_fit') else len(history)
    report['boosting_parameters']=params if config.get('boosting') else None
    # Reporting only, after candidate/guard are frozen. These labels never enter fit,
    # calibration, early stopping, sample selection or promotion decisions.
    import pandas as pd
    reference=pd.read_csv(w/'evaluation/target_window_reference.csv.gz').set_index('window_id')
    report['evaluation']={}
    for role in ('source_test','target_test'):
        with np.load(w/(role+'.npz')) as stored: pack=dict(stored)
        labels=pack['y'] if 'y' in pack else reference.loc[pack['window_id'],'class_id'].to_numpy(dtype='int64')
        report['evaluation'][role]=dict(parent=metrics(labels,m.predict(pack['X_seq'],pack['X_tab'])['probabilities']),candidate=metrics(labels,candidate.predict(pack['X_seq'],pack['X_tab'])['probabilities']))
    write_json(output/'learning_report.json',report);write_json(output/'training_manifest.json',job)
    return report

def worker(root, ledger):
    import threading,traceback
    learning=Learning(root,ledger)
    # One worker is configured in Compose. Interrupted immutable jobs are never promoted.
    with ledger.db() as c:
        c.execute("UPDATE learning_jobs SET status='failed',error='Worker restarted during training; parent retained',finished=? WHERE status='training'",(time.time(),))
        c.execute("UPDATE learning_samples SET state='pending' WHERE state='training'")
    def heartbeat():
        while True:
            with ledger.db() as c:c.execute("INSERT INTO learning_control VALUES('heartbeat',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(time.time()),))
            time.sleep(15)
    threading.Thread(target=heartbeat,daemon=True).start()
    while True:
        job=learning.claim()
        if job:
            print('Learning job',job['id'],job['stage'],len(job['samples']),flush=True)
            try: print('Learning result',learning.finish(job,train_candidate(learning,job)),flush=True)
            except Exception as e:
                traceback.print_exc();learning.finish(job,error=str(e))
        time.sleep(5)
