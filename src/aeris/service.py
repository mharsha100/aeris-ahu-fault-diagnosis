"""Loopback demo API, live metrics and integrity-checked bundle selection."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os,json,time,hashlib,threading,urllib.request
from urllib.parse import urlparse,parse_qs,quote
import numpy as np
import pandas as pd
import torch
from .operations import Ledger,grouped_explanation
from .learning import Learning
from .modeling import load_hybrid,sha256_file
from .workflow import diagnose,canonical_window,SETTINGS

class Service:
    def __init__(self,root):
        self.root=Path(root); self.ledger=Ledger(os.environ.get('AERIS_DB',str(self.root/'data/operator_feedback/ledger.sqlite3'))); self.lock=threading.RLock(); self.model=None; self.error=None; self.identity=None; self.name=None; self.packs={}; self.learning=Learning(self.root,self.ledger); torch.set_num_threads(2)
    def catalog(self):
        return {'models':[dict(version=a['stage'],bundle=a['bundle'],stage='Active '+cohort+' learning stage') for cohort,a in self.learning.status()['active'].items()], 'automatic_selection':True}
    def load(self,requested=None,cohort="genuine",historical=False):
        active=self.learning.active(cohort); name=active['bundle']
        if requested is not None:
            if not historical and requested!=name: raise ValueError('Models are selected automatically from the active learning stage; submit the input once.')
            name=requested
        folder=self.learning.bundle_path(name)
        self.name=name
        identity=(name,sha256_file(folder/'READY.json'),sha256_file(folder/'hybrid.json'),tuple((f,(folder/f).stat().st_mtime_ns,(folder/f).stat().st_size) for f in ['forest.joblib','temporal_state.pt','source_scaler.json','window_protocol.json']))
        # Hash all fitted artifacts on load; a restart rechecks all artifacts.
        if self.model is None or self.identity!=identity:
            self.model=None
            m=load_hybrid(folder); m.forest.n_jobs=1
            if m.configuration.get('feedback_status')=='rejected_do_not_promote':raise ValueError('Rejected feedback candidate cannot be deployed')
            self.model=m; self.identity=identity; self.name=name
            if name==self.learning.source():
                m.configuration['source_architecture_version']=m.configuration.get('model_version')
                m.configuration['model_version']='v0';m.configuration['learning_stage']='source_trained'
        self.error=None
        return self.model
    def ready(self):
        with self.lock:
            try: self.load(); return True
            except Exception as e: self.model=None; self.error=str(e); return False
    def infer(self,body,recorded=False):
        key=body.get('request_id'); cohort=body.get('cohort','genuine')
        if not isinstance(key,str) or not 1<=len(key)<=128: raise ValueError('A stable request_id is required')
        if cohort not in ['genuine','test']: raise ValueError('Invalid cohort')
        digest=hashlib.sha256(json.dumps({'recorded':recorded,**body},sort_keys=True,allow_nan=False).encode()).hexdigest()
        with self.lock:
            prior=self.ledger.lookup(key,digest)
            if prior:
                with self.ledger.db() as c: old=c.execute('SELECT raw FROM predictions WHERE id=?',(prior['prediction_id'],)).fetchone()
                prior['learning']=self.learning.observe(prior,json.loads(old['raw']),prior['result'].get('adaptation_eligible',False),prior['result'].get('input_provenance',''));return prior
            started=time.perf_counter(); setting=body.get('setting','Unknown'); raw=None; config={}
            try:
                try:model=self.load(body.get('bundle'),cohort);config=model.configuration
                except Exception as exc:raise RuntimeError('Bundle integrity or availability failure: '+str(exc)) from exc
                if recorded:
                    role=body.get('role','source_validation')
                    if role not in ['source_validation','target_test','target_adapt','target_feedback']:raise ValueError('Unsupported recording pool')
                    if role not in self.packs:
                        w=self.root/'data/aeris_windows_v1'; p=dict(np.load(w/f'{role}.npz',allow_pickle=False)); meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id').loc[p['window_id']].reset_index(); self.packs[role]=(p,meta)
                    p,meta=self.packs[role]; i=body.get('index',0)
                    if type(i)!=int or not 0<=i<len(meta):raise ValueError('Invalid recording index')
                    row=meta.iloc[i]; raw=canonical_window(p,i,model.scaler); setting=row.season if role.startswith('source') else row.dataset; times=np.arange(15)*60+row.start_time_s
                    recorded_source=role=='source_validation'
                else:
                    raw=np.asarray(body.get('raw'),dtype='float32'); times=body.get('timestamps'); recorded_source=False
                    if times is None:raise ValueError('Fifteen one-minute timestamps are required')
                eligible=self.learning.context(body,recorded,raw,model)
                result=diagnose(model,raw,setting=setting,timestamps=times,recorded_source=recorded_source,validated_adaptation_context=eligible)
                result['adaptation_eligible']=eligible
                result['input_provenance']=('recorded:'+role+':'+str(row.window_id)) if recorded else 'uploaded_verified_context' if eligible else 'uploaded_unverified_context'
                result['unsupported_setting']=not eligible and (setting not in SETTINGS or not recorded_source)
                if 'class_id' in result:
                    z=(raw-np.asarray(model.scaler['mean']))/np.asarray(model.scaler['scale']); result['shift_fraction']=float((np.abs(z[:,:4])>3).mean())
            except (ValueError,TypeError,KeyError,IndexError) as e:
                result={'status':'blocked','reasons':[str(e)]}
            except Exception as e:
                self.model=None; self.error=str(e); result={'status':'error','reasons':['Model unavailable or incompatible. Check bundle integrity and mounted files. '+str(e)]}
            safe_raw=raw.tolist() if isinstance(raw,np.ndarray) and np.isfinite(raw).all() else None
            record=self.ledger.save(key,digest,cohort,setting,self.name or 'unavailable',config,result,safe_raw,time.perf_counter()-started)
            record['learning']=self.learning.observe(record,safe_raw,result.get('adaptation_eligible',False),result.get('input_provenance',''))
            return record

    def window(self,role,index):
        if role not in ['source_validation','target_test','target_adapt','target_feedback']:raise ValueError('Unsupported recording pool')
        with self.lock:
            model=self.load();w=self.root/'data/aeris_windows_v1'
            if role not in self.packs:
                p=dict(np.load(w/f'{role}.npz',allow_pickle=False));meta=pd.read_csv(w/'window_metadata.csv.gz').set_index('window_id').loc[p['window_id']].reset_index();self.packs[role]=(p,meta)
            p,meta=self.packs[role]
            if not 0<=index<len(meta):raise ValueError('Invalid recording index')
            row=meta.iloc[index]
            return dict(raw=canonical_window(p,index,model.scaler).tolist(),timestamps=(np.arange(15)*60+row.start_time_s).tolist(),features=model.protocol['core_features'],window_id=str(row.window_id),setting=row.season if role=='source_validation' else row.dataset,condition_group=row.condition_group,count=len(meta))

    def evidence(self):
        path=self.root/'experiments/operator_redesign_20261117/presentation.json'
        if not path.exists():path=self.root/'experiments/improvement_20261117/presentation.json'
        if path.exists():
            data=json.loads(path.read_text())
            data['status']='Historical architecture comparison below. The live learning stages are V0 source-trained, V1 confident-label adaptation, V2 technician-label adaptation; they are shown separately above.'
            for row in data.get('comparison',[]):
                row['version']=row.get('version','').replace('V0','Last-state architecture').replace('V1','Attention architecture').replace('V2','Combined architecture')
            for section in ('class_metrics','assurance'):
                for row in data.get(section,[]):
                    for key,value in row.items():
                        if isinstance(value,str):row[key]=value.replace('current refined V2','historical source-only combined architecture').replace('V0','Last-state architecture').replace('V1','Attention architecture').replace('V2','Combined architecture')
            data['learning']=self.learning.status()
            assurance=self.root/'experiments/adaptive_lifecycle_20261117'
            plan=assurance/'v2_assurance/art_plan.json'
            if plan.exists() and json.loads(plan.read_text())['model']==self.learning.active('test')['bundle']:
                assurance=assurance/'v2_assurance';data['art_model']='Promoted V2 demonstration model'
            else:data['art_model']='Source-only V0 model'
            if (assurance/'art_summary.csv').exists():data['art_assurance']=pd.read_csv(assurance/'art_summary.csv').to_dict('records')
            if (assurance/'shap_checks.json').exists():data['shap_assurance']=json.loads((assurance/'shap_checks.json').read_text())
            return data
        old=self.root/'experiments/end_to_end_20261116'
        return {'status':'New experiments are running; prior evidence remains available.','comparison':pd.read_csv(old/'branch_comparison.csv').to_dict('records'),'selection':json.loads((old/'selection.json').read_text())}

    def monitoring(self):
        base=os.environ.get('AERIS_PROMETHEUS_URL','http://prometheus:9090')
        queries={'readiness':'aeris_model_ready','app':'aeris_app_up','throughput':'sum(rate(aeris_requests_total[5m]))','p95':'histogram_quantile(0.95, sum by(le)(rate(aeris_inference_seconds_bucket[5m])))','shift':'aeris_source_shift_fraction','scrape':'up{job="aeris"}'}
        out={'updated':time.time(),'summary':{c:self.ledger.summary(c) for c in ['test','genuine']},'signals':{}}
        for key,expr in queries.items():
            try:
                with urllib.request.urlopen(base+'/api/v1/query?query='+quote(expr),timeout=3) as r:values=json.load(r)['data']['result']
                out['signals'][key]=[dict(labels=x['metric'],value=float(x['value'][1]) if np.isfinite(float(x['value'][1])) else None) for x in values]
            except Exception:out['signals'][key]=[]
        try:
            with urllib.request.urlopen(base+'/api/v1/alerts',timeout=3) as r:out['alerts']=json.load(r)['data']['alerts']
        except Exception:out['alerts']=[];out['monitoring_error']='Prometheus is unavailable; check the monitoring service.'
        return out

def app_up():
    # Main interface assets are served by this same HTTP process. Browser behavior
    # is verified separately; the optional Streamlit workspace is not this UI.
    folder=Path(__file__).resolve().parents[2]/'frontend'
    return all((folder/name).is_file() and (folder/name).stat().st_size>100 for name in ['index.html','app.js','style.css'])
def start(root,host='127.0.0.1',port=8001):
    service=Service(root)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def trusted_host(self):
            return urlparse('http://'+self.headers.get('Host','')).hostname in {'127.0.0.1','localhost','::1','app'}
        def reply(self,obj,status=200,content_type='application/json'):
            payload=(json.dumps(obj,allow_nan=False) if content_type=='application/json' else obj).encode(); self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY');self.send_header('Referrer-Policy','same-origin'); self.end_headers(); self.wfile.write(payload)
        def do_GET(self):
            if not self.trusted_host():return self.reply({'error':'This local demonstration does not accept that host'},421)
            try:return self.route_get()
            except (ValueError,KeyError,IndexError) as e:return self.reply({'error':str(e)},400)
            except Exception as e:return self.reply({'error':'Service unavailable: '+str(e)},503)
        def route_get(self):
            route=urlparse(self.path);q=parse_qs(route.query)
            static={'/':('index.html','text/html; charset=utf-8'),'/index.html':('index.html','text/html; charset=utf-8'),'/style.css':('style.css','text/css'),'/app.js':('app.js','application/javascript')}
            static['/contract.js']=('contract.js','application/javascript')
            if route.path in static:
                filename,mime=static[route.path];return self.reply((service.root/'frontend'/filename).read_text(encoding='utf-8'),content_type=mime)
            if route.path=='/learning':return self.reply(service.learning.status())
            if route.path=='/catalog':return self.reply(service.catalog())
            if route.path=='/window':return self.reply(service.window(q.get('role',['source_validation'])[0],int(q.get('index',['0'])[0])))
            if route.path=='/evidence':return self.reply(service.evidence())
            if route.path=='/monitoring':return self.reply(service.monitoring())
            if self.path=='/healthz':return self.reply({'exporter':'up'})
            if self.path=='/readyz':
                ready=service.ready(); up=app_up(); return self.reply({'model_ready':ready,'app_up':up,'bundle':service.name,'error':service.error},200 if ready and up else 503)
            if self.path=='/metrics':return self.reply(service.ledger.metrics(service.ready(),app_up())+service.learning.metrics(),content_type='text/plain; version=0.0.4')
            if self.path=='/summary':return self.reply({k:service.ledger.summary(k) for k in ['genuine','test']})
            self.reply({'error':'not found'},404)
        def do_POST(self):
            try:
                if not self.trusted_host():return self.reply({'error':'This local demonstration does not accept that host'},421)
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('JSON content type is required')
                if self.headers.get('Origin') and urlparse(self.headers['Origin']).netloc!=self.headers.get('Host'):return self.reply({'error':'Cross-origin writes are not allowed'},403)
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=100000:raise ValueError('Request must be 1..100000 bytes')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict):raise ValueError('Expected JSON object')
                if self.path in ['/infer','/recorded']:return self.reply(service.infer(body,self.path=='/recorded'))
                if self.path=='/feedback':return self.reply(service.learning.feedback(body['prediction_id'],body['feedback']))
                if self.path=='/explain':
                    with service.ledger.db() as c:row=c.execute('SELECT * FROM predictions WHERE id=?',(body['prediction_id'],)).fetchone()
                    if not row or json.loads(row['raw']) is None:raise ValueError('No valid input for explanation')
                    with service.lock:
                        model=service.load(row['bundle'],row['cohort'],historical=True)
                        if row['bundle']!=service.name:raise ValueError('Restore the prediction bundle before explaining it')
                        return self.reply(grouped_explanation(model,json.loads(row['raw'])))
                self.reply({'error':'not found'},404)
            except (ValueError,TypeError,KeyError) as e:self.reply({'error':str(e)},400)
            except Exception as e:self.reply({'error':str(e)},500)
    server=ThreadingHTTPServer((host,port),Handler); threading.Thread(target=server.serve_forever,daemon=True).start();return server,service
