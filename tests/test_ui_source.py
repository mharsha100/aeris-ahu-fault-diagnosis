"""Streamlit source-level integration test, not a browser rendering measurement.
Run against the local demo: AERIS_API=http://127.0.0.1:8011 python tests/test_ui_source.py
"""
from pathlib import Path
import os,json,time
import requests,numpy as np
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('AERIS_API','http://127.0.0.1:8011');API=os.environ['AERIS_API']
def summary():return requests.get(API+'/summary',timeout=15).json()['test']
def main():
    before=summary();a=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60);assert not a.exception
    for _ in range(3):a.run(timeout=60)
    assert summary()['eligible_predictions']==before['eligible_predictions']
    start=time.perf_counter();next(b for b in a.button if b.label=='Diagnose').click().run(timeout=60);elapsed=time.perf_counter()-start
    assert not a.exception;record=a.session_state['record'];pid=record['prediction_id']
    for _ in range(3):a.run(timeout=60)
    assert a.session_state['record']['prediction_id']==pid
    assert summary()['eligible_predictions']==before['eligible_predictions']+1
    next(s for s in a.selectbox if s.label=='Agreement').set_value('unsure')
    next(s for s in a.selectbox if s.label=='Satisfaction').set_value(4)
    next(t for t in a.text_area if t.label=='Observation / inspection note').set_value('Automated Streamlit source-level integration test, not genuine inspection.')
    next(b for b in a.button if b.label=='Save review record').click().run(timeout=60);assert not a.exception
    next(b for b in a.button if b.label=='Save review record').click().run(timeout=60);assert not a.exception
    after=summary();assert after['predictions_with_feedback']==before['predictions_with_feedback']+1;assert after['eligible_predictions']==before['eligible_predictions']+1
    result={'test':'Streamlit AppTest server-side widget flow','browser_render_verified':False,'prediction_id':pid,'reruns_do_not_increment_predictions':True,'feedback_duplicates_do_not_increment_coverage':True,'source_level_ui_execution_seconds':elapsed,'before':before,'after':after}
    output=ROOT/'experiments/end_to_end_20261116'/os.environ.get('AERIS_UI_TEST_OUTPUT','ui_source_test.json')
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
