"""Live API verification. Test cohort only; preserves IDs for idempotency checks."""
import argparse,json,time,uuid,urllib.request,urllib.error
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--api',default='http://127.0.0.1:8011');p.add_argument('--output');args=p.parse_args();api=args.api
    def get(path):return json.load(urllib.request.urlopen(api+path,timeout=30))
    def post(path,body):return json.load(urllib.request.urlopen(urllib.request.Request(api+path,json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=60))
    result={'ready':get('/readyz'),'before':get('/summary'),'checks':{}}
    body={'request_id':'test-'+uuid.uuid4().hex,'cohort':'test','role':'source_validation','index':0}
    a=post('/recorded',body);b=post('/recorded',body);assert a['prediction_id']==b['prediction_id'];result['checks']['idempotent_prediction']=True
    assert 'class_id' in a['result'];result['valid_prediction']=a
    f={'prediction_id':a['prediction_id'],'feedback':{'agreement':'unsure','satisfaction':4,'note':'Automated deployment test; not genuine inspection'}}
    post('/feedback',f);post('/feedback',f)
    raw=np.zeros((15,10));raw[:,:4]=20;raw[:,[4,9]]=1
    for case,data in [('unsupported',{'raw':raw.tolist(),'setting':'Unknown','timestamps':list(range(0,900,60))}),('malformed',{'raw':[[1]],'setting':'Unknown','timestamps':[0]}),('edge_command',{'raw':np.where(np.indices(raw.shape)[1]==5,1.1,raw).tolist(),'setting':'Unknown','timestamps':list(range(0,900,60))})]:
        r=post('/infer',dict(request_id=uuid.uuid4().hex,cohort='test',**data));result[case]=r
        assert r['result']['status']==('review_required' if case=='unsupported' else 'blocked')
    result['explanation']=post('/explain',{'prediction_id':a['prediction_id']})
    times=[]
    for i in range(100):
        start=time.perf_counter();r=post('/recorded',dict(request_id=uuid.uuid4().hex,cohort='test',role='source_validation',index=i));times.append(time.perf_counter()-start);assert 'class_id' in r['result']
    result['api_100']={'p50_seconds':float(np.quantile(times,.5)),'p95_seconds':float(np.quantile(times,.95)),'under_5_seconds':sum(t<5 for t in times),'scope':'Warm HTTP request through durable inference service; excludes browser render'}
    result['after']=get('/summary');assert result['after']['test']['predictions_with_feedback']==result['before']['test']['predictions_with_feedback']+1;result['checks']['feedback_deduplication']=True
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
