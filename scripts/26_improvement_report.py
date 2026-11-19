"""Render measured improvement results and presentation figures, after verification."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'experiments/improvement_20261117';FIG=OUT/'figures'
def read(name):return json.loads((OUT/name).read_text())
def table(rows):
    rows=list(rows)
    if not rows:return 'No observations.'
    keys=list(rows[0]);fmt=lambda x: f'{x:.4f}' if isinstance(x,float) else str(x).replace('|','/')
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rows)
def main():
    p=read('presentation.json');selection=read('selection.json');ops=read('operations_verified.json');browser=read('browser_verified.json');ex=read('explanation_checks.json');adv=pd.read_csv(OUT/'adversarial_summary.csv');trials=pd.read_csv(OUT/'release_trials.csv');FIG.mkdir(exist_ok=True)
    failure=read('failure_demo.json');assert failure['fired'] and failure['recovered'] and failure['persistent_feedback']
    plt.rcParams.update({'figure.dpi':160,'savefig.facecolor':'#f7f9f4','axes.facecolor':'#f7f9f4','figure.facecolor':'#f7f9f4','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for j,version in enumerate(['v0','v1','v2']):
        r=read(selection['per_architecture'][version]+'_release_metrics.json');axes[0].bar(j-.16,r['accuracy'],.3,color='#1d6e60');axes[0].bar(j+.16,r['macro_f1'],.3,color='#b47d40')
    axes[0].set(xticks=[0,1,2],xticklabels=['V0','V1','V2'],ylim=(0,1),ylabel='Score',title='Retained source-development models');axes[0].legend(['Accuracy','Macro F1'],loc='lower right')
    for fam,color in [('physics','#1d6e60'),('calibrated_physics','#b47d40')]:
        a=trials[trials.family==fam];groups=[a[a.version==v].macro_f1 for v in ['v0','v1','v2']];axes[1].errorbar(np.arange(3)+(-.05 if fam=='physics' else .05),[x.mean() for x in groups],yerr=[x.std(ddof=0) for x in groups],fmt='o',capsize=4,label=fam.replace('_',' '),color=color)
    axes[1].set(xticks=[0,1,2],xticklabels=['V0','V1','V2'],ylabel='Macro F1 ± seed SD',title='Fresh physical-feature candidates (3 seeds)');axes[1].legend();fig.savefig(FIG/'model_comparison.png');plt.close(fig)
    r=read(selection['bundle']+'_release_metrics.json');matrix=np.array(r['confusion_matrix']);fig,ax=plt.subplots(figsize=(6.5,5.5),layout='constrained');im=ax.imshow(matrix/matrix.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Greens');labels=['Normal','Damper','Cooling valve','Heating valve'];ax.set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,xlabel='Predicted class',ylabel='Reference class',title='Retained V2 · source development\nCounts and row recall');ax.tick_params(axis='x',rotation=20)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]}\n{matrix[i,j]/matrix[i].sum():.1%}',ha='center',va='center',color='white' if matrix[i,j]/matrix[i].sum()>.6 else '#183a2c')
    fig.colorbar(im,ax=ax,label='Fraction of reference class');fig.savefig(FIG/'v2_confusion.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained');x=np.arange(3);ax.bar(x-.18,adv.attack_accuracy,.34,label='Raw robust accuracy',color='#b47d40');ax.bar(x+.18,adv.worst_case_review_rate,.34,label='Review rate (selected worst input)',color='#1d6e60');ax.set(xticks=x,xticklabels=['±0.2°C / ±0.01','±0.5°C / ±0.02','±1°C / ±0.05'],ylim=(0,1.12),ylabel='Fraction of 40 balanced source windows',xlabel='Temperature / command attack budget',title='Bounded randomized attacks: protection depends on review');ax.legend();fig.savefig(FIG/'defensive_review.png');plt.close(fig)
    safety=[]
    for v,n in selection['per_architecture'].items():
        q=read(n+'_release_metrics.json');safety.append(dict(version=v.upper(),faults_predicted_normal=q['fault_to_normal'],fault_support=6468,normal_false_alarms=sum(q['confusion_matrix'][0][1:]),normal_support=462,review_rate=q['review_fraction'],accepted_windows=q['accepted_windows'],accepted_accuracy=q['accepted_accuracy'],accepted_false_normal=q['accepted_false_normal']))
    diagnostics=[]
    for name in ['rf_physics_bal','rf_physics_none','extra_physics','hgb_base_15','hgb_base_31','hgb_physics_15','hgb_physics_31']:
        q=read(name+'/result.json');diagnostics.append(dict(candidate=name,macro_F1=q['macro_f1'],fault_to_normal=q['fault_to_normal']))
    report=f'''# AERIS improvement and readiness report — November 17, 2026

## Outcome

The restored application and monitoring stack are running locally. The latest search completed **nine fresh hybrid models × 50 epochs = 450 epochs**, seven classical diagnostic fits, and nine source-calibration decision fits. The candidate family uses three newly fitted physical-feature RFs (one per seed, shared across pooling comparisons) and nine freshly initialized temporal models. These are additional to November 16's documented 19 temporal runs / 662 epochs. Each architecture still combines Random Forest and its own temporal network. The proposed physical representations did **not** satisfy all preregistered release guards across every seed, so the verified parent models remain deployed. The requested high-accuracy, production-validated model has **not been achieved**. Present this as a working, supervised research system with measured limitations.

The ~69% figure is **macro F1**, not accuracy. Retained V2 source-development accuracy is {r['accuracy']:.2%}, macro F1 {r['macro_f1']:.4f}, macro precision {r['per_class']['macro avg']['precision']:.2%}, and macro recall {r['per_class']['macro avg']['recall']:.2%}. Confidence is calibrated, not forced upward. Unsupported targets always require review.

## What changed and why

1. **Model investigation:** added 46 deterministic physical interactions to the RF's 66 summaries (112 total), including mixing residuals and temperature/command relationships. Added a common physical-context residual head to each distinct temporal architecture. Trained fresh V0 last-state, V1 attention and V2 combined models for 50 completed epochs at seeds 42/43/44. Tested RF, ExtraTrees and histogram gradient boosting as diagnostic comparators. Classical-only diagnostics do not replace the required hybrid.
2. **Controlled calibration:** fitted fusion and class-decision factors on source calibration only. Reviewed macro F1, every class recall, all false-normal predictions and accepted false-normal predictions. Every seed must gain at least 0.01 macro F1 without more missed faults or a recall drop over 0.03. Rejected results remain available; no target-score tuning or relabeling occurred.
3. **Interface:** restored the earlier forest-green/cream design and connected it to the real Python hybrid service. Added raw signal plots, architecture selection, all four probabilities, branch comparison, uncertainty reasons, exact grouped explanations, linked review forms, input CSV validation, and audit downloads. Selections clear stale results; pending requests lock their controls. Model evidence, assurance, monitoring and semester history have dedicated views.
4. **Explainability:** replaced simple group removal with exact four-group Shapley values, evaluating all 16 coalitions through the complete RF + temporal model. The view reports baseline, predicted probability, contributions and additivity residual. Binary fan/occupancy context stays fixed. This explains model behavior relative to a source-mean reference, not physical causation.
5. **Monitoring:** the Prometheus link opens a populated readiness graph, with separate targets/alerts links. Grafana has 26 provisioned panels, explicit health mappings, feedback support and cohort filters, and specific missing-value explanations. Counts come from actual persisted records. No artificial genuine usage was added.
6. **Operations:** kept durable SQLite IDs, feedback history and idempotency; added strict CSV parsing, host/origin and JSON-write boundaries, catalog-only version selection, real-browser checks, and a non-destructive online backup/restore rehearsal. Existing integrity, failure recovery and rollback controls remain active. The legacy Streamlit workspace is still available separately.

## Model measurements

All source-development comparisons use the same 6,930 windows. Later source-day and previously exposed target scores are descriptive, not independent site validation. Target results below were not used to select candidates.

{table(p['comparison'])}

The following source-development counts distinguish raw classifier mistakes from decisions that pass the review gate:

{table(safety)}

### Precision and recall

{table(p['class_metrics'])}

Macro F1 averages the four class F1 scores equally. Accuracy weights every window equally. Macro recall is also balanced accuracy. ECE is ten-bin expected calibration error; lower is better, but binning and sample dependence limit its interpretation. Multiclass Brier is summed squared probability error (range 0–2); lower is better. Log loss for retained V2 is {r['log_loss']:.4f}. None of these metrics proves transfer safety.

### Improvement candidates

{table(p['search'])}

{table(diagnostics)}

The full per-seed rejection reasons are in `release_trials.csv` and `selection.json`. Seed standard deviations describe training variability, not independent-building confidence intervals. Full curves, chosen checkpoints, fusion sweeps, calibration factors, plans and hashes are retained. No epoch was silently skipped; best development checkpoints were selected after all 50 epochs.

## Risk management, robustness and trustworthiness

We use the original M2 risk matrix and [NIST AI RMF Govern, Map, Measure, Manage](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) to organize controls. NIST is a process framework, not a numeric certification score. Implementation uses NumPy/pandas/scikit-learn/PyTorch for reproducible measurement, application validation and review rules, SQLite for traceability, integrity hashes and guarded selection for model changes, and Prometheus/Grafana for operational detection. No risk is declared eliminated merely because a test passed.

{table(p['assurance'])}

### Adversarial test details

{table(adv.to_dict('records'))}

This defensive black-box experiment used 40 source-development windows (10/class), 64 bounded randomized perturbations plus the clean input per window/budget, and regenerated both branches from the same altered raw window. The fault subset has 30 windows. Attack success is conditional on initially correct predictions. Raw robust accuracy counts any discovered misclassification as failure. Accepted-error counts examine all tested candidates under the complete review policy; the displayed worst-input review rate uses the candidate minimizing the true-class probability. Strong review rates represent abstention cost, not robust classification accuracy. These are small, dependent development samples and a finite search, not an adversarial certificate or guarantee against future attacks.

At the source-development review gate, V2 reviews {r['review_fraction']:.2%} of windows, accepts {r['accepted_windows']}, and has {r['accepted_false_normal']} accepted false-normal predictions in this sample. Accepted accuracy is {r['accepted_accuracy']:.2%}. **All {r['normal_windows_reviewed']} normal-reference windows are reviewed and there are {r['accepted_normal_windows']} accepted normal predictions.** The high accepted score therefore does not demonstrate reliable autonomous four-class diagnosis or safe equipment clearance.

### Explanation measurements

Twenty windows (five/class), fixed model and source-mean reference: maximum additivity error **{ex['max_additivity_error']:.2e}**, identical repeatability **{ex['repeatability']:.1%}**, same-class top-group stability under a +0.05°C change **{ex['top_group_stability_under_005C']:.1%}**, mean runtime **{ex['mean_seconds']*1000:.1f} ms**. Exact numerical additivity is expected from exact Shapley enumeration; it is not 100% semantic correctness. Off-manifold reference combinations and correlated sensors remain limitations. Independent technician usefulness has not been measured.

### Why strong data did not automatically yield a high score

The cited [Scientific Data dataset paper](https://www.nature.com/articles/s41597-025-06179-y) describes labeled AHU datasets. It does not guarantee separability for this ten-channel, four-class, blocked-window experiment. Here normal has only 462/6,930 development windows; fault severity and operating conditions affect observability; source and target AHUs differ; and a scenario fault label can persist while its actuator is inactive. The new physical interactions and model comparisons tested plausible improvements without changing the evaluation task.

The observability audit found no exact full-precision duplicate windows with conflicting labels. Quantizing measurements creates some cross-label collisions; that is an illustrative resolution study, **not** proof of a hard ceiling on the original model. More independent operating/fault conditions, sensor-semantic verification and appropriately designed new features are needed before claiming high field accuracy. Randomly mixing neighboring windows or tuning against target labels would make the score less credible.

## Verification and deployment

The main UI is http://127.0.0.1:8510/ ; legacy Streamlit http://127.0.0.1:8511/ ; Grafana http://127.0.0.1:3001/d/aeris-operations ; Prometheus http://127.0.0.1:9091/ . All host ports bind to loopback. This is a local deployment, not cloud production.

Python regression checks: 18 passed. Browser CSV contract checks: 5 passed. All three catalog versions returned valid hybrid predictions. A real browser submitted one demonstration review twice: the ledger contains exactly one prediction, one current feedback record, and one history entry for that action. Restart preserved all summary totals. Backup restore reproduced every table's row count and content hash in an independent temporary database. Grafana queried actual Prometheus values matching SQLite totals; 26 panels and healthy scrape targets were verified.

Operational checks: {', '.join(k for k,v in ops['checks'].items() if v)}. The current release also executed a missing-default-bundle failure: its inference request returned an error without a diagnosis, `AerisUnavailable` fired, restoring the selection cleared the alert and recovered inference, and feedback survived another restart. The 100-window browser test also triggered the review-burden warning; this is an observed workload signal, not an availability failure.

Browser measurements and visual checks:

```json
{json.dumps(browser,indent=2)}
```

The API readiness metric verifies bundle availability/integrity; the UI-availability metric checks served assets. Browser correctness is separately tested. A green readiness indicator is not a measure of diagnostic accuracy.

## Why some metrics have no data

- No recent traffic: five-minute latency quantiles need actual requests. Zero throughput while idle is expected.
- No ratings or agreement responses: the relevant denominator is zero. `unsure` is separate from agree/disagree.
- No genuine independently inspected outcomes: live accuracy, recall and field calibration cannot yet be established. Demonstration feedback cannot fill this gap.
- Insufficient independent cases: the original five-case reporting floor remains unmet. Three shared-weather groups and many related windows are not independent site/fault cases.

## Repository and course evidence

Branch: `aeris-improvements-20261117`; HEAD `961876af8df55a203f3305cfc58a453ff727fec7`. The overall November 16 implementation began from a clean checkout. This November 17 follow-up began with that implementation and evidence still uncommitted and preserved it. Final status remains uncommitted local work; no commit or push was performed. Exact final status and the changed-file inventory are included in `git_status_final.txt` and `changed_files.txt`.

Main follow-up changes: `frontend/`, `src/aeris/physical.py`, `src/aeris/modeling.py`, `src/aeris/operations.py`, `src/aeris/service.py`, scripts 20–27 plus `backup_ledger.py`, frontend/representation tests, Docker/Compose configuration, Grafana provisioning, README and dated documentation. Full source and experiments are in the delivery archive; local data and fitted weights remain in the existing checkout and are intentionally excluded from Git, the image build context and the lightweight source/evidence archive.

Course cross-check: 3.3 → assurance measurements and limitations; 5.3 → preserved M2 risk register plus new controls; 7.6 → matched model, seed, class and partition comparisons; 8.4–8.5 → container, real-browser, persistence and recovery evidence; 9.1–9.4 → actual metrics, support, thresholds, owners and firing/recovered alert; 9.5 → linked feedback with separate demonstration/genuine cohorts. Historical Word submissions remain unchanged. This is a dated, reproducible results addendum.

## Readiness decision and remaining work

**Ready for a supervised local semester demonstration with transparent evidence. Not ready for autonomous field diagnosis or a claim of high production accuracy.** The application has useful production-oriented practices; its prediction quality, independent-case support and operational governance do not yet establish production readiness. Retrospective target macro F1 is weak and target review remains mandatory.

Before wider deployment: acquire independently inspected site outcomes across all classes and operating modes; define acceptance criteria with a qualified operations owner; validate sensor mappings, shift and class coverage; retest adversarial and explanation usefulness on independent cases; implement authenticated role-based access and TLS for any network exposure; establish off-host backup retention and disaster recovery objectives; and evaluate sustained load and availability beyond one laptop. The backup rehearsal here closes one local recovery test, not those remaining requirements.

Never claim every risk is covered or that a high explanation repeatability score proves a correct diagnosis. The semester contribution is the integrated, reproducible pipeline, its executed experiments, its accountable operating controls, and the honest finding that the present data/features/protocol do not yet deliver the desired diagnostic performance.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'docs/RESULTS_20261117.md').write_text(report,encoding='utf-8');print('Report and three presentation figures written.')
if __name__=='__main__':main()
