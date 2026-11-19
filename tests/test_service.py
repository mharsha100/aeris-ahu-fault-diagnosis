import unittest,tempfile,os,sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from aeris.service import Service
ROOT=Path(__file__).resolve().parents[1]
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.old=os.environ.get('AERIS_DB');os.environ['AERIS_DB']=str(Path(self.tmp.name)/'ledger.sqlite3');self.service=Service(ROOT)
    def tearDown(self):
        if self.old is None:os.environ.pop('AERIS_DB',None)
        else:os.environ['AERIS_DB']=self.old
        self.tmp.cleanup()
    def test_same_request_does_not_infer_again(self):
        b=dict(request_id='id',cohort='test',role='source_validation',index=0);a=self.service.infer(b,True);z=self.service.infer(b,True);self.assertIn('class_id',a['result']);self.assertEqual(a,z)
        b['index']=1
        with self.assertRaises(ValueError):self.service.infer(b,True)
    def test_unverified_settings_and_missing_timestamps(self):
        raw=np.zeros((15,10));raw[:,:4]=20;raw[:,[4,9]]=1
        b=dict(request_id='id',cohort='test',raw=raw.tolist(),setting='CoolingSeason',timestamps=list(range(0,900,60)))
        a=self.service.infer(b);self.assertEqual(a['result']['status'],'review_required')
        b['request_id']='new';b.pop('timestamps');self.assertEqual(self.service.infer(b)['result']['status'],'blocked')
    def test_unavailable_model_is_error_not_prediction(self):
        self.service.load=lambda: (_ for _ in ()).throw(ValueError('Corrupted checksum'))
        a=self.service.infer(dict(request_id='id',cohort='test'),True);self.assertEqual(a['result']['status'],'error');self.assertNotIn('class_id',a['result'])
    def test_recovery_reloads_cleared_model_without_restart(self):
        root=Path(self.tmp.name)/'isolated';dest=root/'data/aeris_models/v1_attention';shutil.copytree(ROOT/'data/aeris_models/v1_attention',dest)
        service=Service(root);self.assertTrue(service.ready());selection=root/'data/deployment_selection.json';selection.write_text(json.dumps({'bundle':'missing'}));self.assertFalse(service.ready());selection.write_text(json.dumps({'bundle':'v1_attention'}));self.assertTrue(service.ready());self.assertIsNotNone(service.model)
        raw=np.tile(service.model.scaler['mean'],(15,1));raw[:,[4,9]]=1
        r=service.infer(dict(request_id='recovered',cohort='test',raw=raw.tolist(),timestamps=list(range(0,900,60)),setting='Unknown'))
        self.assertIn('class_id',r['result'])
if __name__=='__main__':unittest.main()
