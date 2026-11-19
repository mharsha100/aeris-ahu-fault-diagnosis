from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dash',ROOT/'scripts/29_build_operator_dashboards.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def main():
 sel='{cohort="$cohort"}'
 stage=d.panel(41,'Which learning stage is active?','state-timeline',(0,0,16,6),[d.target('aeris_learning_stage'+sel,'Active stage')],'0 = original source model; 1 = pseudo-label adaptation; 2 = reviewed-label adaptation. This is a stage, not a quality score. Test and genuine registries are isolated.')
 stage['fieldConfig']['defaults'].update(color={'mode':'fixed','fixedColor':'blue'},mappings=[dict(type='value',options={str(i):dict(text=t,color=c) for i,t,c in [(0,'V0 · source','blue'),(1,'V1 · confident labels','purple'),(2,'V2 · reviewed labels','green')]})]);stage['options']=dict(showValue='always',mergeValues=True,rowHeight=.8,legend={'showLegend':False},tooltip={'mode':'single'})
 health=d.panel(42,'Is the background trainer available?','state-timeline',(16,0,8,6),[d.target('aeris_learning_worker_up','Trainer')],'Heartbeat is independent of the inference service. Training never blocks the HTTP request.')
 health['fieldConfig']['defaults'].update(color={'mode':'fixed','fixedColor':'green'},mappings=[dict(type='value',options={'0':dict(text='Unavailable',color='red'),'1':dict(text='Online',color='green')})]);health['options']=dict(showValue='always',mergeValues=True,legend={'showLegend':False},tooltip={'mode':'single'})
 samples=d.bars(d.panel(43,'What learning data has accumulated?','barchart',(0,6,12,9),[d.target('label_join(aeris_learning_samples'+sel+',"sample_status"," · ","kind","state")','Windows',True,True)],'Unique sensor windows, split by pseudo/reviewed label and state. Eight new or corrected windows start a batch after a short quiet period. Repeated/overlapping windows are not independent field cases.'),'sample_status')
 samples['transformations'][0]['options']['excludeByName'].update(kind=True,state=True,instance=True,job=True)
 jobs=d.trend(d.panel(44,'Are trained candidates being accepted?','timeseries',(12,6,12,9),[d.target('aeris_learning_jobs{cohort="$cohort",status="'+s+'"}',s.title()) for s in ['training','promoted','rejected','failed']],'Persisted counts by outcome. Promoted requires source-retention checks versus both the parent and V0. A retained source score is not proof of target improvement. Review exact metrics and parent lineage in the AERIS workspace.'))
 obj=d.dashboard('aeris-learning','AERIS · Adaptation lifecycle',[stage,health,samples,jobs],[d.link('AERIS learning workspace','http://127.0.0.1:8510/'),d.link('Operator overview','/d/aeris-operations'),d.link('Prometheus targets','http://127.0.0.1:9091/targets')]);obj['time']={'from':'now-30m','to':'now'};dest=ROOT/'deployment/grafana/dashboards';(dest/'aeris-learning.json').write_text(json.dumps(obj,indent=2))
 for name in ['aeris.json','aeris-feedback.json']:
  p=dest/name;dashboard=json.loads(p.read_text());dashboard['links']=[x for x in dashboard['links'] if x['title']!='Adaptation lifecycle']+[d.link('Adaptation lifecycle','/d/aeris-learning')];p.write_text(json.dumps(dashboard,indent=2))
if __name__=='__main__':main()
