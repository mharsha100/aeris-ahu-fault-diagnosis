"""Controlled missing-bundle fault; always restore prior selection in finally."""
from pathlib import Path
import json,time,urllib.request,urllib.error,uuid,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/operator_redesign_20261117';selection=ROOT/'data/deployment_selection.json'
def get(url):
    try:return json.load(urllib.request.urlopen(url,timeout=10))
    except urllib.error.HTTPError as e:return json.load(e)
def prediction():
    body={'request_id':uuid.uuid4().hex,'cohort':'test','role':'source_validation','index':0}
    return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8011/recorded',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=30))
def main():
    prior=selection.read_bytes() if selection.exists() else None
    evidence={'before':get('http://127.0.0.1:8011/readyz'),'feedback_before':get('http://127.0.0.1:8011/summary'),'samples':[]}
    try:
        selection.write_text(json.dumps({'bundle':'missing_controlled_failure_demo'}))
        evidence['failed_inference']=prediction();assert evidence['failed_inference']['result']['status']=='error'
        for i in range(10):
            ready=get('http://127.0.0.1:8011/readyz');alerts=get('http://127.0.0.1:9091/api/v1/alerts');evidence['samples'].append({'phase':'failure','ready':ready,'alerts':alerts,'time':time.time()})
            if any(a['labels']['alertname']=='AerisUnavailable' and a['state']=='firing' for a in alerts['data']['alerts']):break
            time.sleep(5)
        assert not ready['model_ready'];assert any(a['labels']['alertname']=='AerisUnavailable' and a['state']=='firing' for a in alerts['data']['alerts'])
        evidence['fired']=True
    finally:
        if prior is None:selection.unlink(missing_ok=True)
        else:selection.write_bytes(prior)
        (OUT/'failure_demo.json').write_text(json.dumps(evidence,indent=2))
    for i in range(10):
        ready=get('http://127.0.0.1:8011/readyz');alerts=get('http://127.0.0.1:9091/api/v1/alerts');evidence['samples'].append({'phase':'recovery','ready':ready,'alerts':alerts,'time':time.time()})
        if ready['model_ready'] and not any(a['labels']['alertname']=='AerisUnavailable' for a in alerts['data']['alerts']):break
        time.sleep(5)
    evidence['recovered']=ready['model_ready'] and not any(a['labels']['alertname']=='AerisUnavailable' for a in alerts['data']['alerts'])
    evidence['recovered_inference']=prediction();assert 'class_id' in evidence['recovered_inference']['result']
    evidence['feedback_before_restart']=get('http://127.0.0.1:8011/summary')
    subprocess.run(['docker','compose','restart','app'],cwd=ROOT,check=True,capture_output=True)
    for i in range(20):
        try:
            if get('http://127.0.0.1:8011/readyz')['model_ready']:break
        except (OSError,ValueError):pass
        time.sleep(3)
    evidence['feedback_after_restart']=get('http://127.0.0.1:8011/summary')
    evidence['persistent_feedback']=evidence['feedback_before_restart']==evidence['feedback_after_restart']
    evidence['targets']=get('http://127.0.0.1:9091/api/v1/targets')
    evidence['grafana_dashboard']=get('http://127.0.0.1:3001/api/dashboards/uid/aeris-operations')
    for i in range(6):
        evidence['grafana_real_data']=get('http://127.0.0.1:3001/api/datasources/proxy/uid/aeris-prometheus/api/v1/query?query=aeris_eligible_predictions')
        if evidence['grafana_real_data']['data']['result']:break
        time.sleep(5)
    assert evidence['grafana_real_data']['data']['result'], 'Grafana datasource must return actual series after recovery'
    (OUT/'failure_demo.json').write_text(json.dumps(evidence,indent=2));assert evidence['recovered'] and evidence['persistent_feedback'];print('Failure fired, recovered, and feedback survived restart.')
if __name__=='__main__':main()
