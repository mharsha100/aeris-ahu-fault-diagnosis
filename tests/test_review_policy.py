from pathlib import Path
import json,sys,unittest
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.workflow import diagnose
class ReviewPolicy(unittest.TestCase):
    def setUp(self):
        w=ROOT/'data/aeris_windows_v1';sc=json.loads((w/'source_scaler.json').read_text());protocol=json.loads((w/'window_protocol.json').read_text());self.raw=np.tile(sc['mean'],(15,1));self.raw[:,[4,9]]=1
        self.state=dict(k=0,confidence=.95,disagreement=False)
        def predict(seq,tab):
            p=np.full((1,4),(1-self.state['confidence'])/3);p[0,self.state['k']]=self.state['confidence']
            return dict(class_id=np.array([self.state['k']]),confidence=np.array([self.state['confidence']]),branch_disagreement=np.array([self.state['disagreement']]),probabilities=p,forest_probabilities=p,temporal_probabilities=p)
        self.model=SimpleNamespace(scaler=sc,protocol=protocol,configuration={'learning_stage':'initial_source_training'},predict=predict)
    def call(self,**kw):return diagnose(self.model,self.raw,setting=kw.pop('setting','CoolingSeason'),recorded_source=kw.pop('recorded_source',True),timestamps=kw.pop('timestamps',np.arange(15)*60),**kw)
    def test_confidence_and_agreement(self):
        self.assertEqual(self.call()['status'],'research_prediction');self.state['confidence']=.89;self.assertEqual(self.call()['status'],'review_required');self.state['confidence']=.99;self.state['disagreement']=True;self.assertEqual(self.call()['status'],'review_required')
    def test_learning_requires_ninety_five_percent_confidence(self):
        self.state['confidence']=.93
        self.assertEqual(self.call()['status'],'research_prediction')
        self.assertEqual(self.call(setting='SZVAV',recorded_source=False,validated_adaptation_context=True)['status'],'review_required')
        self.state['confidence']=.96
        self.assertEqual(self.call(setting='SZVAV',recorded_source=False,validated_adaptation_context=True)['status'],'research_prediction')
    def test_scope_overrides_confidence(self):
        self.assertEqual(self.call(setting='Unknown')['status'],'review_required');self.assertEqual(self.call(recorded_source=False)['status'],'review_required');self.state['k']=3;self.assertEqual(self.call()['status'],'review_required')
    def test_sensor_and_time_controls(self):
        self.assertEqual(self.call(timestamps=np.arange(15)*61)['status'],'blocked');self.raw[:,4]=0;self.assertEqual(self.call()['status'],'blocked');self.raw[:,4]=1;self.raw[:,7]=1.2;self.assertEqual(self.call()['status'],'blocked')
    def test_feedback_always_review_and_no_actuation(self):
        self.model.configuration['learning_stage']='verified_feedback';r=self.call();self.assertEqual(r['status'],'review_required');self.assertFalse(r['automatic_actuation_allowed'])
if __name__=='__main__':unittest.main()
