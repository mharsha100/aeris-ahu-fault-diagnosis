"""Warm server-render path; deliberately not a browser-paint benchmark."""
from pathlib import Path
import os,time,json
import numpy as np
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('AERIS_API','http://127.0.0.1:8011')
def main():
    a=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60);assert not a.exception
    start=time.perf_counter();next(b for b in a.button if b.label=='Diagnose').click().run(timeout=60);cold=time.perf_counter()-start;assert not a.exception
    times=[];ids=[];result={}
    for i in range(100):
        next(n for n in a.number_input if n.label=='Window index').set_value(i)
        start=time.perf_counter();next(b for b in a.button if b.label=='Diagnose').click().run(timeout=60);times.append(time.perf_counter()-start);assert not a.exception;ids.append(a.session_state['record']['prediction_id'])
        result={'scope':'Streamlit AppTest server-side render and live HTTP inference; excludes browser network and painting','browser_render_verified':False,'cold_first_diagnosis_seconds':cold,'completed':len(times),'seconds':times,'p50':float(np.quantile(times,.5)),'p95':float(np.quantile(times,.95)),'under_five_seconds':sum(t<5 for t in times),'unique_prediction_ids':len(set(ids)),'cohort':'test'}
        (ROOT/'experiments/end_to_end_20261116/ui_server_latency.json').write_text(json.dumps(result,indent=2))
        if (i+1)%20==0:print(f'{i+1} server UI requests; p95={result["p95"]:.3f}s',flush=True)
    assert len(set(ids))==100
if __name__=='__main__':main()
