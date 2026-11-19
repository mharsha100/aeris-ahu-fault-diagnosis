import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aeris.operations import Ledger
from aeris.workflow import diagnose
import numpy as np
class LedgerTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'state.sqlite3';self.db=Ledger(self.path)
    def tearDown(self):self.tmp.cleanup()
    def prediction(self,key='same'):
        return self.db.save(key,'hash','test','Unknown','v0',{},dict(status='review_required',class_id=0,confidence=.9),None,.1)
    def test_duplicate_and_restart(self):
        a=self.prediction();b=self.prediction();self.assertEqual(a['prediction_id'],b['prediction_id']);self.assertEqual(Ledger(self.path).summary('test')['eligible_predictions'],1)
        with self.assertRaises(ValueError):self.db.lookup('same','different')
    def test_feedback_denominators_and_provenance(self):
        p=self.prediction()['prediction_id'];self.db.feedback(p,dict(agreement='unsure',satisfaction=4))
        self.db.feedback(p,dict(agreement='unsure',satisfaction=4));s=self.db.summary('test');self.assertEqual(s['predictions_with_feedback'],1);self.assertIsNone(s['agreement']);self.assertEqual(s['unsure'],1)
        self.assertIsNone(self.db.summary('genuine')['feedback_coverage'])
        with self.assertRaises(ValueError):self.db.feedback(p,dict(verified_class='normal'))
        with self.assertRaises(ValueError):self.db.feedback('unknown',{})
        with self.assertRaises(ValueError):self.db.feedback(p,dict(satisfaction=6))
    def test_metric_labels(self):
        p=self.prediction()['prediction_id'];self.db.feedback(p,dict(note='private technician note'))
        text=self.db.metrics(True,True);self.assertNotIn(p,text);self.assertNotIn('private technician',text)
    def test_malformed_input(self):
        self.assertEqual(diagnose(None,np.zeros((2,3)),setting='Unknown')['status'],'blocked')
        self.assertEqual(diagnose(None,np.full((15,10),np.nan),setting='Unknown')['status'],'blocked')
if __name__=='__main__':unittest.main()
