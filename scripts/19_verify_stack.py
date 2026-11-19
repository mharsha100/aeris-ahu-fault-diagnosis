"""Verify actual rollback, service health, bounded monitoring and Grafana data."""
from pathlib import Path
import json,time,urllib.request,urllib.parse,subprocess,uuid,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/end_to_end_20261116';selection=ROOT/'data/deployment_selection.json'
def get(url):return json.load(urllib.request.urlopen(url,timeout=40))
def infer():
    body={'request_id':uuid.uuid4().hex,'cohort':'test','role':'source_validation','index':1}
    return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8011/recorded',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=60))
def hashes(name):
    folder=ROOT/'data/aeris_models'/name
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
def query(expression,grafana=False):
    base='http://127.0.0.1:3001/api/datasources/proxy/uid/aeris-prometheus' if grafana else 'http://127.0.0.1:9091'
    return get(base+'/api/v1/query?query='+urllib.parse.quote(expression))
def main():
    prior=selection.read_bytes();chosen=json.loads(prior.decode('utf-8-sig'))['bundle'];before={n:hashes(n) for n in [chosen,'v1_attention']};evidence={'selected':chosen}
    try:
        selection.write_text(json.dumps({'bundle':'v1_attention','reason':'Controlled rollback verification'}))
        evidence['rollback_ready']=get('http://127.0.0.1:8011/readyz');evidence['rollback_prediction']=infer()
        assert evidence['rollback_prediction']['bundle']=='v1_attention' and 'class_id' in evidence['rollback_prediction']['result']
    finally:selection.write_bytes(prior)
    evidence['restored_ready']=get('http://127.0.0.1:8011/readyz');evidence['restored_prediction']=infer()
    assert evidence['restored_prediction']['bundle']==chosen and 'class_id' in evidence['restored_prediction']['result']
    evidence['artifact_hashes_unchanged']=before=={n:hashes(n) for n in before};assert evidence['artifact_hashes_unchanged']
    for _ in range(12):
        targets=get('http://127.0.0.1:9091/api/v1/targets')
        if all(t['health']=='up' for t in targets['data']['activeTargets']):break
        time.sleep(5)
    assert targets['data']['activeTargets'] and all(t['health']=='up' for t in targets['data']['activeTargets']);evidence['targets']=targets
    ids=subprocess.check_output(['docker','compose','ps','-q'],cwd=ROOT,text=True).split()
    containers=json.loads(subprocess.check_output(['docker','inspect',*ids],cwd=ROOT,text=True))
    evidence['container_health']=[dict(name=c['Name'],image=c['Config']['Image'],image_id=c['Image'],health=c['State']['Health']['Status']) for c in containers]
    assert all(c['health']=='healthy' for c in evidence['container_health'])
    evidence['dashboard']=get('http://127.0.0.1:3001/api/dashboards/uid/aeris-operations')
    assert len(evidence['dashboard']['dashboard']['panels'])==24
    summary=get('http://127.0.0.1:8011/summary');evidence['feedback_summary']=summary
    expressions=['aeris_eligible_predictions','aeris_predictions_with_feedback','aeris_model_ready','aeris_app_up','aeris_setting_requests_total','aeris_verified_class_support','(aeris_agree{cohort="genuine"}/(aeris_agree{cohort="genuine"}+aeris_disagree{cohort="genuine"})) and ((aeris_agree{cohort="genuine"}+aeris_disagree{cohort="genuine"})>0)']
    # Wait for the next real scrape before checking exact persisted totals.
    for _ in range(6):
        values=query('aeris_eligible_predictions',True)
        actual={r['metric']['cohort']:float(r['value'][1]) for r in values['data']['result']}
        if actual=={k:v['eligible_predictions'] for k,v in summary.items()}:break
        time.sleep(5)
    assert actual=={k:v['eligible_predictions'] for k,v in summary.items()}
    evidence['grafana_queries']={e:query(e,True) for e in expressions}
    if summary['genuine']['agree']+summary['genuine']['disagree']==0:assert evidence['grafana_queries'][expressions[-1]]['data']['result']==[]
    feedback=evidence['grafana_queries']['aeris_predictions_with_feedback']['data']['result']
    assert {r['metric']['cohort']:int(r['value'][1]) for r in feedback}=={k:v['predictions_with_feedback'] for k,v in summary.items()}
    evidence['browser_visual_verified']=False
    (OUT/'stack_final.json').write_text(json.dumps(evidence,indent=2));print('Rollback/restoration, immutable bundles, healthy services, live Grafana totals and missing-data semantics passed.')
if __name__=='__main__':main()
