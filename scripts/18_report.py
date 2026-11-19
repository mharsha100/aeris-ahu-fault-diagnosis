"""Assemble the executed evidence; no model fitting or selection in this report."""
from pathlib import Path
import json,sys,platform
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/end_to_end_20261116'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def table(df,columns=None):
    if columns is not None:df=df[columns]
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '—' if pd.isna(v) else f'{v:.5f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
    selection=read('selection.json');name=selection['bundle'];results=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/result.json'))]
    rows=[];fusion=[]
    for r in results:
        rows.append(dict(bundle=r['name'],version=r['version'],seed=r['seed'],epochs=r['epochs'],best_epoch=r['best_epoch'],device='CUDA',seconds=r['seconds'],rf_seconds=r.get('rf_seconds'),forest_weight=r['selected_weight'],**r['parameters'],macro_f1=r['hybrid']['macro_f1'],fault_to_normal=r['hybrid']['fault_to_normal'],normal_false_alarm=r['hybrid']['normal_false_alarm'],brier=r['hybrid']['brier'],log_loss=r['hybrid']['log_loss'],split='aeris-windows-v1',bundle_sha256=r.get('bundle_sha256'),timing_note=r.get('recovery_note','Measured run wall time; recovered trial timing is incomplete if noted in its result')))
        for f in r['fusion']:fusion.append(dict(bundle=r['name'],**{k:v for k,v in f.items() if k!='recall'}))
    trials=pd.DataFrame(rows);trials.to_csv(OUT/'trial_table.csv',index=False);pd.DataFrame(fusion).to_csv(OUT/'fusion_sweep.csv',index=False)
    robust=[]
    current=pd.read_csv(OUT/'robustness_cases.csv');current['bundle']=name
    for (bundle,case),g in pd.concat([pd.read_csv(OUT/'baseline_robustness_cases.csv'),current]).groupby(['bundle','case']):
        valid=g.predicted.notna();robust.append(dict(bundle=bundle,case=case,windows=len(g),fault_windows=int((g.true_class!=0).sum()),predictions=int(valid.sum()),blocked=int((g.status=='blocked').sum()),review_required=int((g.status=='review_required').sum()),macro_f1=float(f1_score(g.loc[valid,'true_class'],g.loc[valid,'predicted'],labels=[0,1,2,3],average='macro',zero_division=0)) if valid.any() else None,false_normal=int(g.false_normal.sum()),accepted_false_normal=int(g.accepted_false_normal.sum())))
    robust=pd.DataFrame(robust);robust['delta_macro_f1_vs_clean']=robust.apply(lambda r:r.macro_f1-robust.loc[(robust.bundle==r.bundle)&(robust.case=='clean'),'macro_f1'].iloc[0],axis=1);robust.to_csv(OUT/'robustness_comparison.csv',index=False)
    calibration=[]
    for bundle in selection['per_architecture'].values():
        r=next(r for r in results if r['name']==bundle)
        for state,key in [('raw weighted branches','uncalibrated'),('source-calibration temperature scaling','hybrid')]:calibration.append(dict(bundle=bundle,method=state,macro_f1=r[key]['macro_f1'],brier=r[key]['brier'],log_loss=r[key]['log_loss']))
    calibration=pd.DataFrame(calibration);calibration.to_csv(OUT/'calibration_comparison.csv',index=False)
    dec=pd.DataFrame([dict(version=d['version'],anchor_F1=d['anchor']['macro_f1'],finalist_mean=d['finalist_mean'],finalist_std=d['finalist_std'],finalist_worst=d['finalist_worst'],retained=d['retained'],selected=d['selected'],reason='; '.join(d['reasons']) or 'All guards passed') for d in selection['decisions']])
    before=pd.read_csv(OUT/'before_after.csv');post=pd.read_csv(OUT/'postselection_evaluation.csv');branches=pd.read_csv(OUT/'branch_comparison.csv');coverage=pd.read_csv(OUT/'coverage.csv');groups=pd.read_csv(OUT/'subgroups.csv');eff=pd.read_csv(OUT/'efficiency_comparison.csv')
    errors=pd.read_csv(OUT/'selected_errors.csv.gz');explanations=read('explanations.json');audit=read('audit.json');deploy=read('deployment_test_final.json');failure=read('failure_demo.json');ui=read('ui_server_latency.json')
    selected=before.loc[before.name==name].iloc[0];gate=coverage.loc[(coverage.bundle==name)&(coverage.threshold==.9)].iloc[0]
    group_report=groups.loc[(groups.bundle==name)&(groups.group_type=='season'),['group','class_name','windows','recordings','recall','fault_to_normal_rate','review_rate','evidence']]
    snapshot=read('stack_final.json')['feedback_summary'] if (OUT/'stack_final.json').exists() else deploy['after']
    genuine=snapshot['genuine'];test=snapshot['test'];training=read('selected_training_metrics.json')
    text=f'''# AERIS executed results — November 16, 2026

The local Docker demo deploys **{name}**, selected on source development evidence. It is research decision support, not field validation. Every deployed prediction combines RF and TNN. No target score selected hyperparameters, thresholds, or a winner. All reported target results are retrospective. Browser visual verification is pending; API and Streamlit source-level tests are separate evidence.

## Checkout, provenance and scope

Work is in `C:\\MSAIS\\EGN6216-AI SYSTEMS\\AHU_PROJECT\\aeris-data-warmup_updated`, branch `aeris-end-to-end-20261116`, starting at `961876af8df55a203f3305cfc58a453ff727fec7` from `milestone4-final-20261104-194011`. The starting checkout was clean. Changes are uncommitted and have not been pushed. `git_status_final.txt` and `changed_files.txt` capture the final working tree; historical milestone evidence is preserved.

All 47 selected raw inputs were present and hash-matched. Source has 453,645 rows; target has 37,439. No duplicate row IDs, duplicate recording timestamps, nonincreasing timestamps, or non-minute intervals were found. Source has no missing core values; target has 91 missing temperature values. The 470 channel conversion checks passed, and the fitted scaler exactly matched 140,402 eligible source-training rows. Existing prepared windows and frozen splits were reused. There are 27,720 training, 6,930 validation, 6,930 calibration and 6,929 later-source-test windows. Target adaptation/feedback/test have 1,420/1,136/1,136 unlabeled prepared windows.

Source is G36-1wk; targets are SZVAV and SZCAV. No authoritative AHU1/AHU2 numeric alias was found. Blocks do not overlap across partitions, but they reuse recordings and only three shared-weather source conditions. Thousands of windows are not thousands of independent cases. The five-independent-case floor is unmet. No confidence interval based on treating windows as independent is reported. Later-source scores are dependent blocked evaluation, not unseen-condition generalization.

October complete bundles reproduced source-validation macro F1 exactly: V0 0.688665 and V1-attention 0.688413. M4 reports were verified at approximately 0.683975/0.684224/0.686507 source and 0.194156/0.201800/0.181100 retrospective target for V0/V1/V2. Complete M4 forests are absent locally, so those M4 numbers are historical reports, not a fresh exact reproduction. The available download had temporal weights only. The historical V2 adaptation gate selected two windows from one day and one class and skipped an update.

## Executed tuning and model choice

The search executed {len(results)} temporal trials: one two-epoch pilot, three fresh 50-epoch architecture anchors, six ten-epoch screens and nine 50-epoch finalists across seeds 42/43/44. Total completed temporal epochs: {int(trials.epochs.sum())}. Pilot runtime was {next(r['seconds'] for r in results if r['name']=='pilot_v0_s42'):.1f} seconds. The declared runtime budget was 14,400 seconds. Each epoch and completed trial was saved. A cross-device serialization check failed once; saved artifacts were recovered, and serialization was then checked on the same CPU. Separate CPU/CUDA numerical differences and any class/review changes are recorded in `deployment_precision_audit.csv`. Recovered screening wall time is incomplete; do not add its epoch-only time to claim an exact total search duration.

RF varied trees 100/200/300, depth 12/18/24, leaf size 1/3/6, feature subsampling 0.5/0.7/sqrt and balanced/no class weighting. TNN varied width 16/32/48, dropout .05/.15/.30, learning rate .0005/.001/.002, decay .00001/.0001/.001, batch 128/256/512 and weighted/unweighted loss. Both branches remained in fusion weights .25/.50/.75/.90 RF. These were coupled reproducible configurations, not isolated causal effects of each parameter. V0 final state, V1 attention and V2 combined representations were trained fresh. The 29-step theoretical receptive field already spans the 15-sample input, so increasing it was not justified. Existing RF features already include rolling mean/std/min/max/last/trend; no unsupported missing-value imputation was added. The short small-width configuration did not meet diagnostic guards and was not promoted for speed.

The checkpoint rule was best source-development macro F1 with loss tie-break, no early stop. Calibration used only reserved source calibration. Target labels were opened only after selection was written. Predeclared promotion guards require every finalist seed to avoid increased fault-to-normal and accepted false-normal counts, with no class recall loss over .03; mean F1 must beat its matched anchor. Passing families rank by mean F1, then fewer false-normal errors, with fixed seed 42 as representative. Thresholds remain provisional. Rejected families retain their anchor.

{table(dec)}

Full-budget results below use the actual CPU deployment path; the complete 19-trial table preserves CUDA training results, parameters and hashes.

{table(before,['name','seed','macro_f1','fault_to_normal','normal_false_alarm','seconds'])}

Selected source-development macro F1 is **{selected.macro_f1:.6f}**, with **{int(selected.fault_to_normal)}/6,468** fault windows classified normal and **{int(selected.normal_false_alarm)}/462** normal windows assigned a fault. Training macro F1 is {training['macro_f1']:.6f}; the training/development gap remains substantial. The V2 anchor has more pre-review false-normal windows than the V0/V1 anchors (635 versus 596/601); its selection is the declared macro-F1 ranking among retained anchors, not a claim that it wins every safety metric. At the fixed .90 gate, {int(gate.review_windows)}/6,930 windows require review and {int(gate.accepted_false_normal)}/{int(gate.accepted_fault_denominator)} accepted fault windows are classified normal. In this development set all normal windows also require review. Review-or-fault-escalation recall is {gate.review_or_fault_escalation_recall:.4f}; this window statistic does not establish reliable technician outcomes or independent-case performance. Source classification is not evidence of target reliability.

Matched branches:

{table(branches)}

An isolated RF class-weight experiment held the network, RF size/depth, seed and fusion fixed. Balanced weighting gave F1 .688902, 596 fault-to-normal and 255 normal false alarms; removing weighting gave .658811, 200 and 388. Reduced missed faults came with markedly worse normal recognition and macro F1, so balancing was retained. Per-class results are in `weighting_ablation.json`.

Temperature scaling versus raw fusion:

{table(calibration)}

Reliability bins, proper probability scores and the .50–.99 review sweep are saved. Calibration does not guarantee increased F1 or target calibration. The .90 threshold was retained rather than chosen using retrospective target results.

Post-selection descriptive evaluation:

{table(post)}

The historical target macro-F1 goal .70 is not a validated achievement. Target inputs always require review. Source-only parents remain deployed: cross-setting updates are visibly skipped for unsupported applicability, and genuine-feedback updates are skipped because no independently verified, aligned eligible labels exist. Per-window gate rejections and candidate diversity are saved; gates were not loosened to force an update. The explicit reviewed feedback-candidate command enforces genuine provenance, reserved feedback data, case-separated partitions, per-class case floors and parent comparison. This successful-update path is implemented but unexercised with real eligible labels.

## Operational groups, robustness and explanations

{table(group_report)}

Every source group is marked insufficient independent evidence. Empty class/season combinations are unsupported, not zero-accuracy measurements. Target setting/class counts appear in `subgroups.csv`, with 100% mandatory review. An explicit pandas/sklearn group analysis is used as the MetricFrame equivalent because the needed counts and denominators are directly inspectable.

The bounded test uses the same 100 source windows (25 per class, 75 fault windows) for each bundle and six scenarios. Both branches are regenerated from the same modified raw sensor input. Gaussian temperature noise is sigma .2°C clipped to ±.6°C; the supply offset is +1°C; dropout introduces one missing sample; stuck supply holds the first value. No claim of worst-case adversarial robustness is made. F1 is conditional on an actual prediction; all-blocked cases have no F1. The case table also records every status and input ID.

{table(robust)}

The selected model has {int(((errors.confidence>=.9)).sum())} high-confidence development errors, including {int(((errors.confidence>=.9)&errors.false_normal).sum())} high-confidence false-normal errors before application review. The 20 highest-priority errors are saved with recording IDs and disagreement. Explanation tests covered {len(explanations)} representative windows spanning the classes; exact-repeat consistency passed for {sum(e['repeat_identical'] for e in explanations)}. Largest grouped-effect change after +.05°C was {max(e['max_effect_change_under_005C'] for e in explanations):.5f}; mean explanation runtime was {np.mean([e['seconds'] for e in explanations]):.3f} seconds. Full-hybrid group replacement is sensitivity to a training-mean reference, not a physical cause; off-manifold and correlation limits are displayed. Independent technician usefulness remains untested.

## Implemented controls and deployment evidence

NIST Govern/Map/Measure/Manage and the actual M2 risk matrix are mapped in `docs/TRUSTWORTHINESS.md`. Controls are implemented in the application and workflows: strict shape/unit/range/fan/occupancy/timestamp checks; mandatory unsupported-setting review; confidence/agreement/source-bound reasons; integrity-checked bundles; durable idempotent predictions; versioned feedback history; explicit training/promotion; matched selection guards; bounded-label metrics; and tested failure recovery. Original R1–R8 remain visible, with R9 explanation risk added. Field false-normal risk remains Critical; a software test does not reduce its physical consequence.

The app, Prometheus and Grafana use loopback ports 8510, 9091 and 3001; the API is 8011. Docker uses a pinned Python 3.11 CPU image; CUDA training was verified on the RTX 4070 Laptop GPU with PyTorch 2.9.1+cu128. Container GPU passthrough is unverified. A one-epoch V2 CPU container training/reload smoke test completed and is explicitly excluded from model selection. Data and fitted artifacts are mounted, not baked into the image. SQLite, Prometheus and Grafana persist in named volumes.

The final live API test passed valid inference, malformed rejection, physical-bound edge rejection, unknown-setting review, stable prediction IDs, duplicate-feedback protection and full-hybrid explanation. All 15 regression tests passed on Windows and in the Linux container. Of 100 warm HTTP requests, {deploy['api_100']['under_5_seconds']} completed under five seconds; p50 {deploy['api_100']['p50_seconds']:.3f}s, p95 {deploy['api_100']['p95_seconds']:.3f}s. These exclude browser rendering. The separate warm Streamlit AppTest run used 100 requests: p50 {ui['p50']:.3f}s, p95 {ui['p95']:.3f}s, {ui['under_five_seconds']}/100 under five seconds; cold first diagnosis took {ui['cold_first_diagnosis_seconds']:.3f}s. It is server widget execution, not browser-paint evidence. The original 95/100 rendered-UI-under-five-seconds target remains unverified.

{table(eff)}

Actual controlled missing-model alert fired: **{failure['fired']}**; recovery: **{failure['recovered']}**; feedback unchanged across restart: **{failure['persistent_feedback']}**. A cache bug discovered in recovery testing was fixed: a restored bundle must reload when the cached model is absent even if its identity is unchanged. The regression test and actual successful inference before restart both passed. Grafana's provisioned 24-panel dashboard and datasource returned real Prometheus series. Current genuine predictions/feedback/verified labels are {genuine['eligible_predictions']}/{genuine['predictions_with_feedback']}/{genuine['verified_labels']}; test predictions/feedback are {test['eligible_predictions']}/{test['predictions_with_feedback']}. No test opinion is presented as independent ground truth. Missing genuine agreement/satisfaction return no data, not artificial zero quality.

The three priority alerts are availability for 15 seconds; fixed-source temperature shift >20% with at least 20 recent requests for 30 seconds; and review workload >80% with 20 requests in five minutes for one minute. These provisional signals, bounded queries, owners and responses are in `docs/MONITORING_PLAN.md`. No external alert delivery or long-term monitoring is claimed.

## Course submission cross-check and reproducibility

| Master template | Current evidence |
|---|---|
| 3.3 Trustworthiness | Govern/Map/Measure/Manage matrix, implemented guards, subgroup/robustness/explanation results |
| 5.3 Risk | Original M2 matrix, R1–R8 plus R9, initial/residual rationale, owners, triggers and responses |
| 7.6 Model evidence | Trial table/configs/learning curves, three-seed mean/std/worst, branch/ablation/calibration/group/retrospective evaluation |
| 8.4–8.5 Deployment controls/tests | Dockerfile/Compose, integrity/readiness, CPU training smoke, live API and failure/restart tests, pending browser check |
| 9.1–9.5 Monitoring/feedback | Prometheus rules, provisioned Grafana, persistent linked feedback, denominators/cohorts, live alert and recovery evidence |

The professor DockerFiles example and prior Module 8 chat established the three-service layout and five-second scrape workflow. AERIS adds provisioning, integrity/readiness and task-specific alerts. No invented setting is attributed to the professor. The user clarified that the trustworthiness ideas were general class guidance and no separate resource exists; this is not a pending dependency.

Run commands and rollback instructions are in `docs/DEPLOYMENT_RUNBOOK.md`. Complete evidence is in `experiments/end_to_end_20261116`; model bundles remain under ignored `data/aeris_models`. The evidence package excludes raw data, model binaries and per-window prediction arrays. Regeneration uses the existing local data and recorded dependency locks.

## Pending and unverified items

1. Browser visual verification and browser-submitted feedback: automated computer use stopped because its safeguard could not verify the active browser URL. No alternate browser automation was used to bypass it. Open http://127.0.0.1:8510 manually, keep cohort **test**, diagnose a recorded window, save an **unsure** review, and confirm its prediction ID and one added feedback record in /summary and Grafana. Check layout and measure actual browser-to-painted-result latency separately. AppTest exercised the same form code and persistence but does not satisfy this visual check.
2. Independent field performance and verified feedback adaptation: obtain at least five independent cases per relevant class/setting and independently inspected labels with provenance. Use new recordings for prospective evaluation; do not relabel these inspected target data as unseen. Existing material cannot establish target accuracy or a genuine feedback improvement.
3. Exact M4 artifact reproduction: obtain the complete original M4 RF+TNN bundles matching their manifests. Current historical report verification and exact October reproduction are clearly separated. AHU1/AHU2 aliases require an authoritative mapping only if those numeric names are needed for submission.
4. Explanation usefulness: have technicians evaluate the displayed sensitivities and limits on independent cases. Repeatability alone is insufficient.
5. GPU inside Docker is optional and unverified; the verified CPU demo and host CUDA training work. No installation or data redownload is required for the current demo.
'''
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8');(ROOT/'docs/RESULTS_20261116.md').write_text(text,encoding='utf-8')
    print('Wrote trial/fusion/calibration/robustness tables and executed RESULTS report.')
if __name__=='__main__':main()
