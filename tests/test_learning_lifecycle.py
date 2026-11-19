import json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from aeris.service import Service
from aeris.learning import Learning
from aeris.operations import Ledger

ROOT=Path(__file__).resolve().parents[1]
class LearningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.env=patch.dict(os.environ,{'AERIS_DB':str(Path(cls.temp.name)/'ledger.db')});cls.env.start();cls.s=Service(ROOT)
 @classmethod
 def tearDownClass(cls):cls.env.stop();cls.temp.cleanup()
 def test_01_automatic_stage_and_queue(self):
  s=self.s
  for i in [27,28,29,30,31,32,33,36]:
   body=dict(request_id='adapt-'+str(i),cohort='test',role='target_adapt',index=i)
   r=s.infer(body,True);self.assertEqual(r['model_version'],'v0');self.assertEqual(r['learning']['route'],'queued_v1')
   repeated=s.infer(body,True);self.assertEqual(repeated['prediction_id'],r['prediction_id']);self.assertEqual(repeated['learning']['route'],'already_recorded')
  self.assertEqual(sum(x['count'] for x in s.learning.status()['samples']),8)
  self.assertEqual(s.learning.active('genuine')['stage'],'v0')
 def test_02_holdouts_cannot_train(self):
  s=self.s;r=s.infer(dict(request_id='holdout',cohort='test',role='source_validation',index=0),True)
  self.assertEqual(r['learning']['route'],'evaluation_only')
  response=s.learning.feedback(r['prediction_id'],dict(corrected_class='normal',inspector='test reviewer',provenance='test only'))
  self.assertFalse(response['training_eligible'])
  w=s.window('source_validation',0);model=s.load();self.assertFalse(s.learning.context(dict(setting='SZVAV',context_verified=True),False,w['raw'],model))
 def test_03_review_replaces_pseudo_and_snapshot_invalidates(self):
  s=self.s
  with s.ledger.db() as c:c.execute('UPDATE learning_samples SET updated=?',(time.time()-30,))
  job=s.learning.claim();self.assertIsNotNone(job);self.assertEqual(job['stage'],'v1')
  pid=job['samples'][0]['prediction_id']
  answer=s.learning.feedback(pid,dict(corrected_class='normal',inspector='demonstration reviewer',provenance='simulated test label'))
  self.assertTrue(answer['training_eligible'])
  with s.ledger.db() as c:
   self.assertTrue(json.loads(c.execute('SELECT body FROM feedback WHERE prediction_id=?',(pid,)).fetchone()[0])['training_eligible'])
   history=c.execute('SELECT COUNT(*) FROM feedback_history WHERE prediction_id=?',(pid,)).fetchone()[0]
  s.learning.feedback(pid,dict(corrected_class='normal',inspector='demonstration reviewer',provenance='simulated test label'))
  with s.ledger.db() as c:self.assertEqual(c.execute('SELECT COUNT(*) FROM feedback_history WHERE prediction_id=?',(pid,)).fetchone()[0],history)
  with self.assertRaises(ValueError):s.learning.feedback(pid,dict(corrected_class='normal'))
  self.assertEqual(s.learning.finish(job,dict(guard={'passed':True})),'failed')
  self.assertEqual(s.learning.active('test')['stage'],'v0')
 def test_04_manual_architecture_switch_rejected(self):
  with self.assertRaises(ValueError):self.s.load('boosted_v1_s42','test')
 def test_05_shap_explains_complete_model(self):
  from aeris.operations import grouped_explanation
  m=self.s.load();w=self.s.window('source_validation',0);e=grouped_explanation(m,w['raw'])
  self.assertEqual(e['library'],'shap');self.assertLess(e['additivity_error'],1e-6)
  self.assertEqual(len(e['values']),4)
if __name__=='__main__':unittest.main()
