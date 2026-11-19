"""Live persistence, input boundary, backup restoration and monitoring checks."""
from pathlib import Path
import json,time,urllib.request,urllib.error,urllib.parse,subprocess,uuid
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/operator_redesign_20261117';API='http://127.0.0.1:8011'
def request(path,body=None,headers=None):
    h={'Content-Type':'application/json'};h.update(headers or {});r=urllib.request.Request(API+path,json.dumps(body).encode() if body is not None else None,h)
    try:
        with urllib.request.urlopen(r,timeout=60) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
def get(url):return json.load(urllib.request.urlopen(url,timeout=30))
def docker(*args):return subprocess.check_output(['docker','compose',*args],cwd=ROOT,text=True)
def main():
    for _ in range(45):
        try:
            if request('/readyz')[0]==200:break
        except (OSError,ValueError):pass
        time.sleep(2)
    else:raise RuntimeError('Application did not become ready')
    evidence={'checks':{}};checks=evidence['checks'];evidence['ready']=request('/readyz')[1]
    for name,path,body,headers,expected in [('cross_origin','/feedback',{}, {'Origin':'https://untrusted.invalid'},403),('text_form','/feedback',{}, {'Content-Type':'text/plain'},400),('untrusted_host','/summary',None,{'Host':'untrusted.invalid:8011'},421),('path_traversal','/%2e%2e/AGENTS.md',None,{},404),('negative_index','/window?index=-1',None,{},400)]:
        status,data=request(path,body,headers);assert status==expected,(name,status,data);checks[name]=status
    catalog=request('/catalog')[1];evidence['catalog']=catalog;predictions=[]
    for model in catalog['models']:
        body=dict(request_id='nov17-'+uuid.uuid4().hex,cohort='test',role='source_validation',index=1,bundle=model['bundle']);status,p=request('/recorded',body);assert status==200 and 'class_id' in p['result'] and p['bundle']==model['bundle'];assert request('/recorded',body)[1]['prediction_id']==p['prediction_id'];predictions.append(p)
    evidence['version_predictions']=predictions;checks['all_catalog_versions_and_idempotency']=True
    pid='01def67810d449008547b82416c5d10c'
    code="import sqlite3,json; c=sqlite3.connect('/state/ledger.sqlite3'); p='"+pid+"'; print(json.dumps({t:c.execute('SELECT COUNT(*) FROM '+t+' WHERE '+('id' if t=='predictions' else 'prediction_id')+'=?',(p,)).fetchone()[0] for t in ['predictions','feedback','feedback_history']}))"
    evidence['browser_feedback_counts']=json.loads(docker('exec','-T','app','python','-c',code));assert evidence['browser_feedback_counts']=={'predictions':1,'feedback':1,'feedback_history':1}
    evidence['backup_restore']=json.loads(docker('exec','-T','app','python','scripts/backup_ledger.py','--destination','/state/backups/ledger-'+uuid.uuid4().hex+'.sqlite3'));checks['backup_restore']=True
    before=request('/summary')[1];docker('restart','app')
    for _ in range(45):
        try:
            status,data=request('/readyz')
            if status==200:break
        except (OSError,ValueError):pass
        time.sleep(2)
    else:raise RuntimeError('Application did not recover after restart')
    after=request('/summary')[1];assert before==after;checks['restart_preserves_all_feedback_totals']=True;evidence['summary']=after
    evidence['dashboard']=get('http://127.0.0.1:3001/api/dashboards/uid/aeris-operations');assert len(evidence['dashboard']['dashboard']['panels'])==7;checks['grafana_7_panels']=True
    for _ in range(12):
        targets=get('http://127.0.0.1:9091/api/v1/targets')
        if targets['data']['activeTargets'] and all(t['health']=='up' for t in targets['data']['activeTargets']):break
        time.sleep(3)
    else:raise AssertionError('Prometheus scrape did not recover after application restart')
    evidence['targets']=targets
    evidence['feedback_dashboard']=get('http://127.0.0.1:3001/api/dashboards/uid/aeris-feedback');assert len(evidence['feedback_dashboard']['dashboard']['panels'])==4
    evidence['panel_queries']=[]
    for dashboard in [evidence['dashboard'],evidence['feedback_dashboard']]:
        for panel in dashboard['dashboard']['panels']:
            for target in panel['targets']:
                for cohort in ['test','genuine']:
                    expr=target['expr'].replace('$cohort',cohort).replace('$__range','24h').replace('$__rate_interval','1m')
                    q=get('http://127.0.0.1:3001/api/datasources/proxy/uid/aeris-prometheus/api/v1/query?query='+urllib.parse.quote(expr));assert q['status']=='success'
                    evidence['panel_queries'].append(dict(panel=panel['title'],cohort=cohort,series=len(q['data']['result'])))
    checks['all_panel_queries_both_cohorts']=True
    expression='aeris_eligible_predictions' 
    for _ in range(10):
        q=get('http://127.0.0.1:3001/api/datasources/proxy/uid/aeris-prometheus/api/v1/query?query='+urllib.parse.quote(expression));actual={r['metric']['cohort']:int(r['value'][1]) for r in q['data']['result']}
        if actual=={k:v['eligible_predictions'] for k,v in after.items()}:break
        time.sleep(3)
    else:raise AssertionError('Grafana totals did not match persisted ledger')
    checks['grafana_live_totals_match_sqlite']=True;evidence['grafana_query']=q
    assert after['genuine']['eligible_predictions']==0 and after['genuine']['verified_labels']==0;checks['test_data_not_presented_as_genuine']=True
    ids=docker('ps','-q').split();containers=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True));evidence['health']=[dict(name=c['Name'],image=c['Image'],health=c['State']['Health']['Status']) for c in containers];assert all(c['health']=='healthy' for c in evidence['health'])
    (OUT/'operations_verified.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
