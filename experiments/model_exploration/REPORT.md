# AERIS: First Model Exploration

Measured development evidence for EGN 6216, October 26, 2026. Organized Python scripts are used instead of a new training notebook; the outputs below can be read without executing the code.

## 1. Role of AI, candidate approaches, and selected stage

The AI component classifies an occupied, fan-on AHU interval as normal, damper fault, cooling-valve fault, or heating-valve fault from sensor temperatures and control signals. It returns a probability vector, a most-likely condition, a confidence score and branch disagreement. A deterministic rule routes uncertain results to review and recommends escalation for confident fault predictions; people retain the maintenance decision.

Candidates considered are a Random Forest on window summaries, a causal temporal convolutional network (TCN) on sensor sequences, and their hybrid. The selected first exploration is the supervised source-domain diagnostic hybrid, with the two branches also inspected individually in V0. RF provides a modest tabular comparator and TCN represents temporal patterns; the combined implementation fits the available laptop and project direction. The proposed V1 attention variant tests a richer temporal summary. It is a new architecture trained from scratch, not inherited V0 weights. The versions, cross-setting adaptation, verified feedback, comparisons and eventual UI belong to one AERIS project. The present measured comparison covers initial source training of V0 and V1, not completed transfer or V2.

The non-AI stages are file/schema checks, documented unit conversion, quality and operating-mode filtering, time/day assignment, windowing, deterministic summary features, probability postprocessing and the review/escalation rule. Standardization learns statistics only from source training. Temperature calibration is a fitted postprocessing stage using a separate source-calibration partition; it does not establish target-domain confidence reliability. The operating-scope checks and saved/reloaded inference interface run before the diagnostic output is accepted.

## 2. Inputs, preparation, versions and separation

Source: the G36-1wk simulated AHU dataset, a documented selection of 45 CSV runs from the cooling, heating and shoulder seasons. Target: real LBNL-2019-2 SZVAV/SZCAV records prepared for subsequent cross-setting work. Source and target datasets are attributed in the repository README; both releases are documented as CC BY 4.0. No pretrained weights or paid API are used.

Preparation version is `aeris-data-v1`; the frozen window protocol is `aeris-windows-v1`. The [selected-file manifest](data_audit/input_manifest.csv), [quality audit](data_audit/quality_audit.csv), [feature map](data_audit/feature_mapping.csv), [data contract](data_audit/data_contract.json), [split manifest](data_audit/split_manifest.csv), [window protocol](data_audit/window_protocol.json), and the SHA-256 records in each training run identify the actual inputs. The same three source-pack hashes are present in both model runs.

There are 453,645 source rows, all valid for the ten core channels; 245,702 meet the occupied/fan-on scope. Target preparation found 37,439 rows, of which 37,348 are core-valid and 18,720 are eligible occupied/fan-on observations. The 91 invalid target rows fall outside that active operating scope. Missing, nonnumeric, nonfinite and out-of-range values are flagged, with no statistical imputation or clipping; invalid rows are excluded from eligible windows and break continuity. Duplicate or unsorted timestamps cause a stop for review rather than silent repair. Source temperatures are converted from Kelvin to Celsius and target temperatures from Fahrenheit to Celsius. Controls are documented fractions. Labels, filenames, timestamps, run IDs and simulation-only true actuator positions do not enter the feature arrays.

Each neural input is `(15 minutes, 10 channels)` at one-minute resolution, with a five-minute window stride. The channels are supply, outdoor, mixed and return air temperatures, fan status, fan-speed command, outdoor-air damper command, cooling-valve command, heating-valve command and occupancy. Each RF input has 66 features: six statistics per channel plus six temperature-difference summaries. Source-train-only normalization is fitted on 140,402 training rows; RF summaries remain in their documented physical units.

| Source partition | Windows | Day groups | Use |
|---|---:|---:|---|
| Training | 27,720 | 12 | Fit RF and TCN |
| Validation | 6,930 | 3 | Checkpoints, fusion selection and development comparison |
| Calibration | 6,930 | 3 | Probability-temperature fitting |
| Final test | 6,929 | 3 | Reserved; not opened by these experiments |

| Class | Training windows | Validation windows |
|---|---:|---:|
| Normal | 1,848 | 462 |
| Damper fault | 12,936 | 3,234 |
| Cooling-valve fault | 8,624 | 2,156 |
| Heating-valve fault | 4,312 | 1,078 |

The actual source split blocks weather days across all scenarios before windowing: first four days per season for training, fifth for validation, sixth for calibration, seventh for test. Eligible rows and windows do not cross partition boundaries. This is **new-day validation within the same simulated scenarios**, not scenario-held-out or building-held-out validation. It narrows the earlier blueprint's scenario/run separation goal and must not be presented as having fulfilled that stronger independence requirement. Overlapping windows within a partition remain dependent; the 6,930 windows are not independent fault injections. Three validation day groups provide limited evidence.

Target allocation is retrospective and documented before model evaluation: 1,420 adaptation, 1,136 feedback and 1,136 final-test windows. The target-input packs omit labels. Neither target observations nor target labels are used for the two initial training runs or this homework's source-development comparison. Final source and target test inputs remain unused. The frozen protocol contains historical V1/V2 adaptation-role wording; its data allocation is preserved for reproducibility, while the architecture version meaning is governed by AGENTS.md and the current training documentation.

## 3. Initial configuration and completed runs

Both runs use seed 42 and freshly fit a 200-tree RF with maximum depth 18, minimum leaf size 3, `sqrt` feature sampling and `balanced_subsample` class weights. Both neural encoders use three causal residual blocks, 32 channels, kernel size 3, dilations 1/2/4, dropout 0.15 and per-time-step channel LayerNorm. AdamW starts at learning rate 0.001 and weight decay 0.0001, with batch size 256, class-weighted cross-entropy, gradient clipping at 1.0 and a validation-driven learning-rate scheduler.

V0 uses the final encoded time step. Its maximum budget was 50 epochs with patience 8; it completed 37 epochs and saved epoch 29. V1 adds learned attention pooling over the observed window. It completed 50 epochs with early stopping disabled and saved epoch 18. The parameter counts are 17,380 and 18,468, respectively. Both RF fits start fresh; identical RF settings, data and seed can yield identical forest predictions without reusing the old weights.

The hybrid mixes RF and temporal probabilities. The same candidate RF weights (0.25, 0.50, 0.75) and source-validation selection rule were used for both runs. V0 selected 0.75 RF / 0.25 TCN; V1 selected 0.50 / 0.50. Separate source calibration fits the probability temperatures. The provisional review rule is confidence below 0.90 **or** branch disagreement. Both save/reload checks passed, with reported maximum probability difference approximately 2.22e-16.

Training environment: Python 3.11.9, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.7.2, joblib 1.5.2 and PyTorch 2.9.1+cu128, on an RTX 4070 Laptop GPU with about 8 GB VRAM. The exact dependency list is in [requirements.txt](../../requirements.txt); GPU and package records are saved per run. No new package was required for attention pooling or evidence collection.

## 4. Evaluation and practical constraints

Macro F1 is primary because normal windows are much less frequent than damper-fault windows and each supported class matters. Per-class precision/recall/F1 and confusion matrices expose missed faults and false alarms. Accuracy, balanced accuracy, log loss and calibration error are supplementary. These are source-validation development scores used in selection, not final generalization or real-target transfer scores.

| Run | Configuration / change | Epochs completed; best | Validation N | Macro F1 | Accuracy | CPU inference p95 |
|---|---|---|---:|---:|---:|---:|
| V0 | RF + causal TCN, final-step pooling; RF fusion 0.75 | 37; 29 | 6,930 | 0.688665 | 76.10% | 22.28 ms |
| V1 | Fresh RF + TCN with attention pooling; RF fusion 0.50 | 50; 18 | 6,930 | 0.688413 | 74.66% | 16.63 ms |

| Class | V0 recall | V1 recall | V0 F1 | V1 F1 |
|---|---:|---:|---:|---:|
| normal | 0.4632 | 0.5931 | 0.3295 | 0.3317 |
| damper_fault | 0.8250 | 0.7647 | 0.8031 | 0.7942 |
| cooling_valve_fault | 0.7050 | 0.7226 | 0.8021 | 0.8064 |
| heating_valve_fault | 0.8089 | 0.8061 | 0.8199 | 0.8214 |

V0's RF-only macro F1 was 0.688434 and its TCN-only macro F1 was 0.667366 in the original training report; the hybrid's improvement over RF alone is very small. V1's selected TCN checkpoint scored approximately 0.6700 in its training history. Neither run establishes that hybrid complexity is superior to simpler alternatives.

Measured RF fit times were 7.35 s (V0) and 6.47 s (V1). Summed per-epoch neural training/validation times were 127.91 s and 197.29 s. These sums exclude setup, final calibration and serialization; they are not full end-to-end wall-clock durations.

CPU timing used the **same 100 validation window IDs in the same order**, deterministic seed 42, 25 windows from each class, three warmup requests, batch size one, four torch threads and RF `n_jobs=1`. Python/platform/processor details and every timed request are saved in each `inference_timing.json` and `inference_timing_samples.csv`. Median latencies were 15.68 ms (V0) and 14.23 ms (V1); p95 was 22.28 ms and 16.63 ms. All 100 measured inference requests per model took less than five seconds. The one-pass timing difference is not a robust architectural speed claim.

Section 2.1's provisional target is that at least 95 of 100 **full local UI requests** render within five seconds. This measurement includes model input checks, inference, calibration and decision rules, but excludes file/model loading, raw preprocessing and UI rendering. It is useful partial latency evidence; the full UI target remains unverified. Likewise, the provisional target-domain quality targets are not assessed by this source-only comparison.

## 5. Inspection of specific failures

Ten errors from each model were inspected. Selection first sorts incorrect predictions by confidence, diversifies by true/predicted class, run and day, then keeps ten. This intentionally inspects relatively confident mistakes, not a random sample of all errors. The complete IDs and metadata appear in each error CSV. Short prefixes below identify those same windows.

### V0: ten reviewed errors

| Window ID prefix | Season | True condition | Predicted | Confidence | Needs review | Branch disagreement |
|---|---|---|---|---:|---|---|
| `0cf0800a4efb` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.8068 | True | False |
| `dfa4aad727bf` | CoolingSeason | cooling_valve_fault | damper_fault | 0.7764 | True | True |
| `3f14833b3384` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.7674 | True | False |
| `9313765cb9be` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.7614 | True | False |
| `a9196f2a21a4` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.7533 | True | False |
| `b6d33352232c` | ShoulderSeason | normal | damper_fault | 0.7442 | True | False |
| `399056f88239` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.7309 | True | False |
| `681ccbbdcc01` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.7308 | True | False |
| `5ce3c34511e0` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.7295 | True | False |
| `4b4047f4ef8f` | HeatingSeason | heating_valve_fault | damper_fault | 0.6906 | True | False |

### V1: ten reviewed errors

| Window ID prefix | Season | True condition | Predicted | Confidence | Needs review | Branch disagreement |
|---|---|---|---|---:|---|---|
| `0f003d964b73` | HeatingSeason | normal | damper_fault | 0.8544 | True | False |
| `4cc1049922cd` | HeatingSeason | heating_valve_fault | damper_fault | 0.8544 | True | False |
| `9e25c457bc45` | HeatingSeason | heating_valve_fault | damper_fault | 0.8544 | True | False |
| `dfa4aad727bf` | CoolingSeason | cooling_valve_fault | damper_fault | 0.8405 | True | False |
| `aca20ace1a23` | HeatingSeason | heating_valve_fault | damper_fault | 0.8274 | True | False |
| `ac3be24fa8cc` | ShoulderSeason | cooling_valve_fault | damper_fault | 0.6820 | True | False |
| `039aa7408963` | ShoulderSeason | cooling_valve_fault | normal | 0.6284 | True | False |
| `c174b5c86e76` | ShoulderSeason | damper_fault | normal | 0.6284 | True | False |
| `0e250aad63e5` | ShoulderSeason | damper_fault | normal | 0.6284 | True | False |
| `9ccf1b6cfe7c` | ShoulderSeason | cooling_valve_fault | normal | 0.6284 | True | False |

Observed patterns and implications:

1. V0 window `0cf0800a4efb...` is a shoulder-season cooling fault classified as a damper fault at confidence 0.8068. Both branches agree but are wrong. V0 shows 397 cooling-to-damper confusions across validation, so branch agreement alone cannot verify a pseudo-label.
2. V0 window `b6d33352232c...` is normal shoulder-season operation classified as a damper fault at 0.7442. There are 173 normal-to-damper confusions; normal precision is only 0.2557 and recall 0.4632.
3. V0 window `4b4047f4ef8f...` is a heating fault classified as a damper fault at 0.6906. This shows the confusion is not limited to cooling cases.
4. V1 winter windows `0f003d964b73...` (normal) and `4cc1049922cd...` (heating fault) are both classified as damper faults at about 0.8544, despite branch agreement. Similar probabilities do not establish the physical cause; raw trend inspection is a next step.
5. The **same** cooling-season window `dfa4aad727bf...` is misclassified as damper fault by both versions. Confidence rises from 0.7764 to 0.8405, while the prediction remains wrong. Higher confidence is not sufficient evidence of improvement.
6. V1 windows `039aa7408963...` (cooling fault) and `c174b5c86e76...` (damper fault) are classified as normal at about 0.6284. Across validation, fault-to-normal predictions increase from 623 to 916. These are raw class-prediction errors; they are not measured technician-escalation failures because the review rule also intervenes.

All 20 inspected mistakes are below the 0.90 threshold and route to review. Across the full validation set there are no incorrect hybrid predictions at confidence >=0.90 for either model. Review fractions are 47.91% for V0 and 45.34% for V1. This observed behavior does not prove confidence is reliable on unseen buildings or the real target setting, and the high review burden remains a limitation. The error-count tables describe all misclassifications, while the selected examples are dependent windows from only a few weather-day groups.

## 6. Targeted change, interpretation and next steps

The targeted architectural change was replacing final-step TCN pooling with learned attention over all 15 observed steps. The hypothesis was that retaining information throughout the window could help distinguish weak normal and fault patterns. RF parameters, core channels, features, data packs, normalization, encoder width/dilations, optimizer starting settings, seed and evaluation examples were held fixed. The fusion-selection procedure was fixed, but selected different weights. **Early stopping/training budget also differed** (37 versus 50 completed epochs), as explicitly recorded. Therefore this is an initial system-configuration comparison, not an isolated causal test of attention.

V1 normal recall increases by about 12.99 percentage points, but normal precision falls (0.2557 to 0.2303). Damper recall falls from 0.8250 to 0.7647. Macro F1 changes from 0.688665 to 0.688413 (delta -0.000252), essentially unchanged in this single-seed development comparison, while accuracy falls by 1.44 percentage points. V1 makes 1,756 errors versus 1,656 for V0. The change does not show an overall diagnostic improvement and does not justify replacing V0 on these results. No significance claim is made from overlapping windows or a single seed.

Next work is to compare under matched training budgets and controlled fusion, inspect the raw sensor traces for the reviewed confusions and false-normal cases, and evaluate at the day/scenario level. Longer context and better-balanced informative normal cases are hypotheses to test, not established remedies. V0/V1/V2 architecture development, cross-setting adaptation, technician-verified feedback and the UI remain part of the same AERIS plan. Their later evidence must be reported without opening final tests during tuning. The earlier legacy adaptation run skipped its update; it is not evidence of a trained V1 architecture or successful transfer. The corrected adaptation code is present but no new target-adaptation result is claimed by this submission.

## Reproduction and evidence navigation

1. Use the existing Python 3.11.9 environment and install `requirements.txt`.
2. Dataset access remains documented in the existing `playground.ipynb` and README dataset links. Keep downloaded data local under `data/`.
3. Run `scripts/01_inventory.py`, `02_prepare_data.py` and `03_make_splits_windows.py` for the documented contracts. Existing prepared packs should be reused for the recorded comparison.
4. Initial V0 command: `python scripts/04_train_v0.py --device cuda --epochs 50 --patience 8`.
5. Revised V1 command: `python scripts/05_train_v1.py --device cuda --epochs 50 --patience 0`.
6. Run `python scripts/07_compare_development.py` and `python scripts/08_submission_evidence.py --models v0 v1_attention --requests 100`.

Use the explicit aeris-data interpreter path from [training_v1.md](../../docs/training_v1.md) on the configured Windows laptop. Existing model outputs are protected; reproducing fresh fits requires a fresh working copy/data-output location or deliberately managed backups. GPU results and measured timing can vary with hardware/runtime. This submission uses organized scripts with persisted JSON/CSV outputs; a newly executed notebook is not required to inspect the results.

Evidence files: [experiment table](experiment_results.csv), [V0 metrics](v0/source_validation_metrics.json), [V1 metrics](v1_attention/source_validation_metrics.json), [V0 errors](v0/error_examples.csv), [V1 errors](v1_attention/error_examples.csv), [V0 timing](v0/inference_timing.json), [V1 timing](v1_attention/inference_timing.json), per-run training histories/configurations and the data audit folder. The root README attributes both dataset releases. No raw dataset, credentials, model weights or paid services are included in the submission files.

## Assignment coverage

| Requirement | Evidence |
|---|---|
| AI role, candidates, selected stage and non-AI stages | Section 1 |
| Data readiness, class balance, preprocessing and separation | Section 2 and data_audit/ |
| Completed initial end-to-end model and configuration | Section 3, V0 training records and inference evidence |
| Metric, sample count and development results | Section 4 and both metrics JSON files |
| 5–10 specific errors and confidence behavior | Section 5, ten examples per model |
| Relevant measured practical constraint vs target | Section 4, CPU timing and stated UI measurement boundary |
| Targeted change, rerun and interpretation | Section 6, V1 training records and comparison |
| Readable outputs, source attribution and reproduction | Saved CSV/JSON, scripts, requirements and root README |
| Git commit and accessible submission links | Repository commit supplied in the submission text |

These sections also provide material for Section 7 of the project template. The template document itself has not been edited as part of this code/report submission.
