# Current AERIS learning lifecycle release

Current image: `aeris-local:20261117c2`. V0 means source-trained, V1 means confident
pseudo-label adaptation, and V2 means technician-label adaptation. One input uses
the active stage automatically. Demonstration V2 was actually trained and promoted;
genuine use retains its separate V0 registry. Historical architecture labels below
are separate experiments and do not describe the current V0/V1/V2 workflow.

Read the [corrected explanation and demo guide](docs/ADAPTIVE_WORKFLOW_20261117.md)
and [executed lifecycle evidence](experiments/adaptive_lifecycle_20261117/).
ART 1.20.1 HopSkipJump attacks the complete predictor; SHAP 0.51.0 ExactExplainer
explains it. Docker runs the app, background learner, Prometheus and Grafana.
Grafana has a 7-panel operations overview, a separate 4-panel feedback view and a
4-panel adaptation view. All metric labels distinguish test and genuine cohorts.

Target macro F1: V0 21.31%, V1 21.70%, V2 25.73%. These are retrospective results,
with simulated technician labels for the demonstration. The workflow is functional;
target diagnostic quality remains limited and is not production-validated.

The dated release notes below are preserved historical records.

## November 17 operator and boosted-model release

Current local image: `aeris-local:20261117b`; default: `boosted_v2_s42`. Seven operator panels plus a separate four-panel feedback dashboard. [Measured results and rollback](docs/RESULTS_OPERATOR_20261117.md). Source gains are modest; transfer and independent field validation remain limited.

# AERIS — AHU fault diagnosis

## November 17 improvement release

The active work is on `aeris-improvements-20261117`. Read the [new results](docs/RESULTS_20261117.md), [demonstration and recovery runbook](docs/RUNBOOK_20261117.md), and [executed evidence](experiments/improvement_20261117/).

The restored main interface at http://127.0.0.1:8510 combines the earlier visual design with real Python hybrid inference, persistent feedback, exact grouped Shapley explanations, model evidence, assurance and monitoring views. The legacy Streamlit workspace is at http://127.0.0.1:8511. Grafana has 26 provisioned panels; the UI links to a populated Prometheus readiness graph, scrape targets and alerts.

Nine additional fresh 50-epoch physical-feature hybrids, seven classical diagnostics and nine calibration fits were evaluated without target-score tuning. Candidates must pass every-seed F1, class-recall and missed-fault gates. The latest report separates software and operational improvements from diagnostic quality; no production accuracy or successful target-transfer claim is made. Real-browser form, visual, monitoring and latency evidence is in the new release directory. Historical reports below remain dated records.
## November 16 local implementation and deployment

The new work is on `aeris-end-to-end-20261116`. Start with the
[Windows runbook](docs/DEPLOYMENT_RUNBOOK.md), [current evidence](experiments/end_to_end_20261116/),
[executed results and limitations](docs/RESULTS_20261116.md),
[trustworthiness/risk update](docs/TRUSTWORTHINESS.md), and
[verified course reference](docs/COURSE_REFERENCE.md). Historical milestone claims
below are preserved and are not claims about the current live deployment.

Run `docker compose up -d --build` from this checkout. The CPU demo is at
http://127.0.0.1:8510, Prometheus at http://127.0.0.1:9091 and Grafana at
http://127.0.0.1:3001/d/aeris-operations. Local fitted bundles and prepared inputs
must exist under `data/`; they are intentionally outside Git and the image.
Normal startup never trains. Use `python deployment/serve.py` for the host
service plus UI; `streamlit run app.py` alone now requires the inference service.

Predictions and optional technician feedback are linked in SQLite. Requests are
explicit and idempotent; widget reruns do not count as new inference. Test and
genuine cohorts are separate. No opinion, unverified correction or simulated
reference label automatically enters training. Genuine reviewed candidate
training is explicit in scripts/14_feedback_candidate.py and never promotes itself.

New trials are fresh source-trained RF+TCN hybrids, with V0/V1/V2 architecture
identities preserved, matched frozen partitions, and saved configs/curves.
Model selection uses source development data only. Later-source and target
scores are descriptive; target labels were historically exposed. Independent
case coverage remains insufficient for new-setting claims. The deployed system
is an offline research demonstration and does not control equipment.

All 19 trials completed. The three tuned families failed the multi-seed safety
guards, so their anchors are retained. The selected V2 anchor has source
development macro F1 0.689975 and retrospective target macro F1 0.224315;
this is not successful target transfer. Source development review coverage,
missed-fault counts and the weak independent-case support are reported together.
Browser visual verification remains pending after the computer-use safeguard
stopped automation; live API and Streamlit server-side tests are recorded separately.

AERIS studies a hybrid Random Forest and temporal neural network for AHU diagnosis,
confidence handling and human review. V0, V1 and V2 denote evolving architectures;
cross-setting adaptation, technician-verified feedback, comparison and the operator
interface belong to the same project.

## Current implementation checkpoint — November2,2026

The M4 implementation is in [milestone4/](milestone4/README.md): three fresh50-epoch
hybrids, gated cross-setting adaptation, simulated verified-reference feedback,
comparison/error analysis and a working Streamlit plus browser interface. Actual
completion gates are recorded in [pipeline status](milestone4/evidence/pipeline_status.json),
[real results](milestone4/evidence/results.csv), [20-case error review](milestone4/evidence/error_review_20.csv)
and [validation](milestone4/evidence/validation_report.json). The Word submission
has not been filled as part of this implementation task.

Current runs are Linux Python3.11.17 CPU runs, separate from the October Windows
GPU exploration below. The root requirements now pin the portable CPU+Streamlit
environment; `requirements-cuda.txt` preserves the CUDA alternative. For the
runnable Streamlit interface, install the current requirements and run
`python -m streamlit run app.py` with the packaged trained bundles and input packs.

These experiments describe selected recordings with only three shared-weather
conditions. Original m3-v1 and its five-independent-case floor are preserved.
All target scores are retrospective/exploratory because labels were exposed in
previous work. Skipped/rejected adaptation is reported as such. Published-reference
feedback is explicitly simulated, and no confidence score authorizes AHU actuation.

The current session cannot push to the original GitHub repository (read-only
integration/no CLI credentials). The release preserves exact source and Git history;
publication of the new original-repository branch remains an access limitation.

## First Model Exploration — October 26, 2026

**Start with the [complete assignment report](experiments/model_exploration/REPORT.md).**
It covers all six homework steps and includes actual configurations, specific
mistakes, practical timing measurements, interpretation and limitations.

- [Measured experiment table](experiments/model_exploration/experiment_results.csv)
- [V0 evidence](experiments/model_exploration/v0/)
- [V1 attention evidence](experiments/model_exploration/v1_attention/)
- [Preparation/split audit](experiments/model_exploration/data_audit/)
- [Organized experiment scripts](scripts/)
- [Text-box submission draft](experiments/model_exploration/SUBMISSION_DRAFT.md)

| Run | Source-validation windows | Macro F1 | Accuracy | CPU inference p95 |
|---|---:|---:|---:|---:|
| V0: RF + final-step TCN | 6,930 | 0.688665 | 76.10% | 22.28 ms |
| V1: fresh RF + attention TCN | 6,930 | 0.688413 | 74.66% | 16.63 ms |

V1 completed 50 neural epochs, retaining checkpoint 18; V0 completed 37, retaining
29. V1 raises normal recall but lowers accuracy and does not improve overall macro
F1. Training budgets and selected fusion weights differ, so this is an initial
system comparison, not an isolated test of attention. CPU timing uses the same
100 development windows and excludes loading, preprocessing and UI rendering.
The full UI latency target and real-target transfer quality remain unverified.

## Implementation and reproduction

The October historical runs used Python3.11.9 and PyTorch2.9.1+cu128 on the
Windows RTX4070 laptop. Those pins are retained in [requirements-cuda.txt](requirements-cuda.txt). The [training guide](docs/training_v1.md) provides explicit
PowerShell commands and operating-scope details.

```powershell
& "C:\Users\mhars\miniconda3\envs\aeris-data\python.exe" -m pip install -r requirements.txt
```

| Stage | Entry point |
|---|---|
| Extracted-file inventory | scripts/01_inventory.py |
| Documented channel/label preparation | scripts/02_prepare_data.py |
| Day-blocked splits and 15-minute windows | scripts/03_make_splits_windows.py |
| Initial hybrid V0 | scripts/04_train_v0.py |
| Fresh attention hybrid V1 | scripts/05_train_v1.py |
| Cross-setting adaptation of selected architecture | scripts/06_cross_adapt.py |
| Saved source-development comparison | scripts/07_compare_development.py |
| Errors and CPU timing evidence | scripts/08_submission_evidence.py |

Both initial runs use 27,720 source-training windows, 6,930 source-validation and
6,930 separate source-calibration windows. Final source/target test packs remain
unused during this exploration. The implemented source split holds out new weather
days in the same scenarios, not entirely unseen buildings/fault injections.
Training outputs are local under data/aeris_models/. Existing bundles are protected
against accidental replacement. The legacy 05_adapt_v1.py command is retired; its
skipped update is not a trained architecture V1 or proof of successful transfer.

## Dataset attribution and local storage

- Simulated source: Ghalamsiah et al., *Labeled Datasets for Air Handling Units
  Operating in Faulted and Fault-free States*, Figshare article 29297999,
  [DOI 10.6084/m9.figshare.29297999](https://doi.org/10.6084/m9.figshare.29297999).
- Real target: Lin and Mitchell (2019), *Data Sets for Evaluation of Building Fault
  Detection and Diagnostics Algorithms*, OEDI,
  [DOI 10.25984/1824861](https://doi.org/10.25984/1824861).

Both dataset releases are documented as CC BY 4.0 in the original access audit;
retain attribution when reusing them. See the executed [playground.ipynb](playground.ipynb),
[Milestone 3 ingestion evidence](milestone3/evidence/) and
[historical warm-up notes](docs/data_warmup_original.md). The current report's
file hashes/manifests identify the selected inputs used for these experiments.

Downloaded datasets, prepared arrays, model weights and local backups stay under
Git-ignored data/. Only code and small audit/model-experiment outputs are included
here. No credentials, paid API or live BMS control is required. This is an offline
research diagnostic aid, not autonomous maintenance control.
