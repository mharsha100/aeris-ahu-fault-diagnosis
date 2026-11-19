"""Durable prediction/feedback ledger. IDs and notes never become metric labels."""
from pathlib import Path
import sqlite3, json, uuid, time, hashlib
from datetime import datetime, timezone
from contextlib import contextmanager
import numpy as np
from .modeling import CLASS_NAMES

def utc(): return datetime.now(timezone.utc).isoformat()
class Ledger:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.db() as c:
            c.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS predictions(id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, input_hash TEXT NOT NULL, created TEXT NOT NULL, cohort TEXT NOT NULL, setting TEXT NOT NULL, bundle TEXT NOT NULL, model_version TEXT NOT NULL, stage TEXT NOT NULL, result TEXT NOT NULL, raw TEXT, seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback(prediction_id TEXT PRIMARY KEY REFERENCES predictions(id), updated TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback_history(id INTEGER PRIMARY KEY, prediction_id TEXT NOT NULL, updated TEXT NOT NULL, body TEXT NOT NULL);
            ''')
    @contextmanager
    def db(self):
        c=sqlite3.connect(self.path,timeout=30); c.row_factory=sqlite3.Row; c.execute('PRAGMA foreign_keys=ON')
        try:
            yield c; c.commit()
        except Exception: c.rollback(); raise
        finally: c.close()
    def lookup(self,key,digest):
        with self.db() as c: row=c.execute('SELECT * FROM predictions WHERE request_key=?',(key,)).fetchone()
        if row and row['input_hash']!=digest: raise ValueError('Request ID already belongs to different input')
        return self.decode(row) if row else None
    def decode(self,row): return dict(prediction_id=row['id'],created=row['created'],bundle=row['bundle'],model_version=row['model_version'],stage=row['stage'],setting=row['setting'],cohort=row['cohort'],seconds=row['seconds'],result=json.loads(row['result']))
    def save(self,key,digest,cohort,setting,bundle,model,result,raw,seconds):
        if cohort not in ('test','genuine'): raise ValueError('Invalid cohort')
        setting=setting if setting in ['CoolingSeason','HeatingSeason','ShoulderSeason','SZVAV','SZCAV'] else 'Unknown'
        with self.db() as c:
            c.execute('INSERT OR IGNORE INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(uuid.uuid4().hex,key,digest,utc(),cohort,setting,bundle,model.get('model_version','unavailable'),model.get('learning_stage','unavailable'),json.dumps(result),json.dumps(raw),seconds))
        return self.lookup(key,digest)
    def feedback(self,pid,body,training_eligible=False):
        if body.get('agreement') not in ['agree','disagree','unsure',None]: raise ValueError('Invalid agreement')
        s=body.get('satisfaction')
        if s is not None and (type(s)!=int or not 1<=s<=5): raise ValueError('Satisfaction must be 1 to 5 or absent')
        for field in ['corrected_class','verified_class']:
            if body.get(field) not in CLASS_NAMES+[None]: raise ValueError('Invalid class')
        if body.get('corrected_decision') not in [None,'review_required','escalate_for_review','research_prediction']: raise ValueError('Invalid correction')
        if body.get('verified_class') and not (body.get('inspector','').strip() and body.get('provenance','').strip()): raise ValueError('Inspection identity and independent evidence provenance are required')
        clean={k:body.get(k) for k in ['agreement','satisfaction','corrected_class','corrected_decision','verified_class','inspector','provenance','note']}
        if any(len(str(v))>4000 for v in clean.values()): raise ValueError('Feedback field too long')
        clean['training_eligible']=bool(training_eligible); clean['verification_state']='eligible_for_learning' if training_eligible else 'inspection_recorded_pending_alignment' if clean['verified_class'] else 'unverified'
        encoded=json.dumps(clean,sort_keys=True)
        with self.db() as c:
            if not c.execute('SELECT 1 FROM predictions WHERE id=?',(pid,)).fetchone(): raise ValueError('Unknown prediction ID')
            old=c.execute('SELECT body FROM feedback WHERE prediction_id=?',(pid,)).fetchone()
            if not old or old['body']!=encoded:
                c.execute('INSERT INTO feedback VALUES(?,?,?) ON CONFLICT(prediction_id) DO UPDATE SET updated=excluded.updated, body=excluded.body',(pid,utc(),encoded))
                c.execute('INSERT INTO feedback_history(prediction_id,updated,body) VALUES(?,?,?)',(pid,utc(),encoded))
        return clean
    def summary(self,cohort='genuine'):
        with self.db() as c: rows=c.execute('SELECT p.*,f.body FROM predictions p LEFT JOIN feedback f ON p.id=f.prediction_id WHERE cohort=? ORDER BY created',(cohort,)).fetchall()
        eligible=[r for r in rows if 'class_id' in json.loads(r['result'])]
        f=[json.loads(r['body']) for r in eligible if r['body']]
        agree=sum(x['agreement']=='agree' for x in f); disagree=sum(x['agreement']=='disagree' for x in f)
        scores=[x['satisfaction'] for x in f if x['satisfaction'] is not None]
        verified=[r for r in eligible if r['body'] and json.loads(r['body']).get('verified_class')]
        overrides=sum(bool(x['corrected_class'] or x['corrected_decision']) for x in f)
        return dict(cohort=cohort,time_window='all persisted records since deployment ledger creation',eligible_predictions=len(eligible),predictions_with_feedback=len(f),feedback_coverage=len(f)/len(eligible) if eligible else None,agree=agree,disagree=disagree,unsure=sum(x['agreement']=='unsure' for x in f),agreement=agree/(agree+disagree) if agree+disagree else None,reviewed_predictions=len(f),overrides=overrides,override_rate=overrides/len(f) if f else None,satisfaction_responses=len(scores),satisfaction_sum=sum(scores),satisfaction_mean=float(np.mean(scores)) if scores else None,verified_labels=len(verified),verified_label_coverage=len(verified)/len(eligible) if eligible else None,verified_quality_status='insufficient independent-case evidence; no operational accuracy claim',genuine_ground_truth=False,ground_truth_status='Cohort selection alone does not establish ground truth; inspection records require independent alignment and case support')
    def metrics(self,ready,app_up):
        lines=['# HELP aeris_model_ready Selected bundle passes integrity and reload checks.','# TYPE aeris_model_ready gauge',f'aeris_model_ready {int(ready)}','# TYPE aeris_app_up gauge',f'aeris_app_up {int(app_up)}']
        with self.db() as c: rows=c.execute('SELECT p.*,f.body FROM predictions p LEFT JOIN feedback f ON p.id=f.prediction_id').fetchall()
        for cohort in ['test','genuine']:
            selected=[r for r in rows if r['cohort']==cohort]; lab='{cohort="'+cohort+'"}'
            results=[json.loads(r['result']) for r in selected]
            def metric(n,v): lines.append(f'aeris_{n}{lab} {v}')
            metric('requests_total',len(selected)); metric('errors_total',sum(x['status']=='error' for x in results)); metric('blocked_total',sum(x['status']=='blocked' for x in results)); metric('review_total',sum(x['status']=='review_required' for x in results))
            metric('unsupported_total',sum(x.get('unsupported_setting',False) for x in results))
            for b in [.01,.025,.05,.1,.25,.5,1,2.5,5,10,float('inf')]:
                bound='+Inf' if b==float('inf') else str(b)
                lines.append(f'aeris_inference_seconds_bucket{{cohort="{cohort}",le="{bound}"}} {sum(r["seconds"]<=b for r in selected)}')
            metric('inference_seconds_count',len(selected)); metric('inference_seconds_sum',sum(r['seconds'] for r in selected))
            latest=[x for x in results if 'confidence' in x][-100:]
            if latest:
                metric('confidence_mean',np.mean([x['confidence'] for x in latest])); metric('source_shift_fraction',np.mean([x.get('shift_fraction',0) for x in latest]))
            metric('shift_support',len(latest))
            s=self.summary(cohort)
            for n in ['eligible_predictions','predictions_with_feedback','agree','disagree','unsure','reviewed_predictions','overrides','satisfaction_responses','satisfaction_sum','verified_labels']: metric(n,s[n])
            for setting in ['CoolingSeason','HeatingSeason','ShoulderSeason','SZVAV','SZCAV','Unknown']:
                group=[r for r in selected if r['setting']==setting]
                lines.append(f'aeris_setting_requests_total{{cohort="{cohort}",setting="{setting}"}} {len(group)}')
                lines.append(f'aeris_setting_reviews_total{{cohort="{cohort}",setting="{setting}"}} {sum(json.loads(r["result"])["status"]=="review_required" for r in group)}')
                verified=[r for r in group if r['body'] and json.loads(r['body']).get('verified_class')]
                for label in CLASS_NAMES:
                    supported=[r for r in verified if json.loads(r['body'])['verified_class']==label]
                    lines.append(f'aeris_verified_class_support{{cohort="{cohort}",setting="{setting}",class_name="{label}"}} {len(supported)}')
                    lines.append(f'aeris_verified_class_correct{{cohort="{cohort}",setting="{setting}",class_name="{label}"}} {sum(json.loads(r["result"]).get("class_name")==label for r in supported)}')
            for version in ['v0','v1','v2','unavailable']:
                lines.append(f'aeris_version_requests_total{{cohort="{cohort}",version="{version}"}} {sum(r["model_version"]==version for r in selected)}')
        return '\n'.join(lines)+'\n'

def grouped_explanation(model,raw,class_id=None):
    """SHAP ExactExplainer applied to the complete predictor's four sensor groups."""
    import shap
    raw=np.asarray(raw,dtype='float32')
    if raw.shape!=(15,10) or not np.isfinite(raw).all(): raise ValueError('Invalid explanation window')
    groups=[('temperatures',[0,1,2,3]),('ventilation_commands',[5,6]),('cooling_command',[7]),('heating_command',[8])]
    pred=model.predict_raw(raw[None])['probabilities'][0]
    k=int(pred.argmax()) if class_id is None else int(class_id)
    if not 0<=k<4: raise ValueError('Explanation class must be in 0..3')
    def coalition(bits):
        windows=np.repeat(raw[None],len(bits),axis=0)
        for j,(_,cols) in enumerate(groups):
            for i in np.flatnonzero(bits[:,j]<.5):windows[i][:,cols]=np.asarray(model.scaler['mean'])[cols]
        return model.predict_raw(windows)['probabilities'][:,k]
    explainer=shap.ExactExplainer(coalition,np.zeros((1,4)),feature_names=[g[0] for g in groups])
    explanation=explainer(np.ones((1,4)),max_evals=16)
    values=[dict(group=g[0],shapley_value=float(explanation.values[0,i])) for i,g in enumerate(groups)]
    base=float(explanation.base_values[0]);prob=float(pred[k])
    return dict(method='SHAP ExactExplainer on the complete RF + temporal + CatBoost predictor',library='shap',library_version=shap.__version__,class_name=CLASS_NAMES[k],values=values,coalitions_evaluated=16,reference_probability=base,prediction_probability=prob,additivity_error=float(abs(sum(v['shapley_value'] for v in values)-(prob-base))),limitation='Shows how four sensor groups move this model probability relative to source-training means. Fan and occupancy remain fixed. This explains the model, not a physical cause. Numerical additivity is not proof of human usefulness; reference combinations may be physically unusual.')
