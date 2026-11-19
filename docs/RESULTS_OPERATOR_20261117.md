# AERIS operator and model release — November 17, 2026

The running local release is `aeris-local:20261117b`, default `boosted_v2_s42`. This follow-up improved all three source models, replaced the 26-panel number wall with seven monitoring panels plus a separate four-panel feedback view, fixed the Prometheus destination, and exposed actual dataset support. It is a supervised research demonstration, not a validated autonomous field diagnostic product.

## What improved in the model

Same frozen source-development partition, 6,930 windows:

| Version | Prior macro F1 | Refined macro F1 | Refined accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|
| V0 | 68.89% | 70.62% | 78.02% | 70.12% | 72.21% |
| V1 | 68.95% | 70.66% | 78.04% | 70.15% | 72.27% |
| V2 | 69.00% | 70.67% | 78.07% | 70.17% | 72.29% |

V2 later-source-day macro F1 improved from 65.42% to **68.20%**, with **75.96% accuracy** over 6,929 windows. This later-source day was evaluated after freezing the current refinement. It shares source context and is not an independent-building evaluation. V2 source-development fault-to-normal mistakes fell from 635 to **523 (17.64% fewer)**. These raw errors still exist; review is not counted as a corrected prediction.

V2 source-development class results:

| Condition | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| normal | 29.42% | 47.19% | 36.24% | 462 |
| damper fault | 83.02% | 80.27% | 81.62% | 3234 |
| cooling valve fault | 86.38% | 79.13% | 82.60% | 2156 |
| heating valve fault | 81.88% | 82.56% | 82.22% | 1078 |

Normal detection is still weak. Macro F1 gives each class equal weight; ordinary accuracy is 78.07%, not 70.67%. Raising confidence without improving correctness would make the system less trustworthy.

V2 calibration: ECE **0.02508** versus 0.03071 previously, multiclass Brier **0.25895** versus 0.27903 (both lower is better). Later-source ECE is **0.04305**, worse than the prior 0.03447: calibration did not improve everywhere. Retrospective target macro F1 fell from 22.43% to **21.31%**; all target requests remain review-required. This is not a transfer improvement.

## What was actually trained

Twelve CatBoost GPU fits completed **1,200 iterations each**, 14,400 total: eight representation/depth/weighting screens, two confirming booster seeds, and two additional-sensor studies. Best source-validation checkpoints were retained; there was no early termination of the 1,200-iteration fits. No target data or labels were used for fitting or current hyperparameter selection.

The deployed representation combines physical interactions from 66 tabular summaries with all 150 values in the 15-minute, 10-channel normalized sequence: 262 features. CatBoost 1.2.10 uses the selected depth 8 and square-root class balancing. Probabilities are temperature-calibrated on source calibration; source development chooses constrained fusion and normal-class factor 1.2. The booster contributes **90%**, and the original calibrated RF–temporal hybrid **10%**, before the final factors/calibration. Both original branches retain nonzero influence.

V0, V1 attention, and V2 combined temporal parents are the distinct, previously trained 50-epoch models. They were not freshly retrained in this follow-up. Booster seeds 42/43/44 passed the fixed source-retention guards for all three parents; this is booster-seed confirmation, not three new independent neural trainings. Seed 42 is deployed consistently. The six-model selector retains all three prior baselines for comparison and rollback. Source-development tuning results must not be presented as an unbiased final holdout estimate.

The separate seven-extra-sensor study reached **75.36% / 74.83% macro F1** (seeds 42/43); seed 42 accuracy was 81.11%. It adds supply-temperature setpoint, three airflow measurements, duct pressure and its setpoint, and outdoor wet-bulb temperature. It excludes true actuator positions, labels, timestamps and file identifiers as features. These results suggest useful information beyond the common ten-channel contract. They are source-only, have no validated target/serving contract, and **are not deployed**. They are not interchangeable with the production input schema or an algorithm-only comparison.

## Risk, robustness, and explainability

The original boost-only gate allowed 7 of 40 windows to produce at least one accepted wrong class at the largest tested perturbation. That finding is preserved in `adversarial_before_parent_gate.csv`. The revised gate requires the retained RF–temporal parent to agree confidently as well as the final model, and preserves input, source-range, setting and branch-agreement checks. It never actuates equipment. Policy: `m4-research-review-v3-parent-retention`.

With the revised gate, the repeated bounded attack suite found **zero accepted wrong classes and zero accepted false-normal cases** at each budget. The suite uses 40 windows (10/class), 65 candidates/window/budget, temperature budgets ±0.2/0.5/1.0°C and command budgets ±0.01/0.02/0.05. Fan/occupancy and timestamps stay fixed. Raw worst-candidate accuracy is **60%, 57.5%, 50%**, and worst-candidate review is **82.5%, 80%, 95%**. The largest-budget raw robustness is worse than the prior 57.5%. These are bounded randomized attacks, not exhaustive adversarial guarantees. Because the policy was tightened after this suite exposed failures, its rerun is a regression check, not an independent safety validation.

On source validation, **50.19%** require review. The remaining 3452 predictions were all correct in that partition; this selective result is **not 100% four-class accuracy** and excludes all normal-reference windows. On the later source day, 59.63% require review, and 2797 accepted predictions were correct. More independent sites and genuine inspections are needed before operational thresholds can be validated.

Exact grouped Shapley explanations evaluate the **whole refined hybrid** with 4 sensor groups and 16 coalitions. Twenty balanced windows had maximum additivity error **1.11e-16**, **100% repeatability**, and **90% top-group stability** under +0.05°C perturbation (prior 95%). Additivity means contributions sum to the prediction change relative to source means. It does not establish causality, technician comprehension, or that explanations are useful; human usefulness remains unmeasured. NumPy/SciPy implement and check the calculation, with PyTorch/scikit-learn/CatBoost running the complete model.

Risk management also includes finite/range/time input checks, integrity hashes, applicability review, origin/host request checks, linked SQLite feedback with idempotency, tested backups and recovery, Prometheus alerts, and versioned model selection. There is no single honest “risk-management percentage” that certifies all of these.

## Why the settings chart looked skewed

Your screenshot counted **application requests**, mostly the CoolingSeason windows selected by earlier latency tests. It did not count training examples. The previous title and red count coloring were misleading.

Actual source-training season counts are **9,240 CoolingSeason, 9,240 HeatingSeason, 9,240 ShoulderSeason**. Source validation and calibration each have **2,310 per season**. Fault classes are unequal: normal 1,848 (6.67%), damper 12,936 (46.67%), cooling 8,624 (31.11%), heating 4,312 (15.56%) in training. This is a real class-support limitation, separate from seasonal balance and unrelated to traffic coloring. Square-root class weighting and macro/per-class metrics address evaluation/training imbalance; they do not manufacture independent cases.

The new settings chart says “Which settings were analyzed?” and uses neutral blue bars. Dataset support is available separately in AERIS Model evidence. The latest 100-request verification sampled evenly across the frozen source-validation index range and records its exact indices; it is explicitly demonstration traffic, not a simulation of genuine operators. Earlier traffic was preserved rather than reset to make the chart look balanced.

## Monitoring design and how to use it

Open [Operator overview](http://127.0.0.1:3001/d/aeris-operations). The seven panels answer seven operational questions:

1. **Health timeline:** model readiness, UI asset availability, and scrape UP/DOWN over time. Green UP/red DOWN; gaps mean no observation.
2. **Request-outcome trend:** reviews, research estimates, rejected inputs and service errors per minute. It shows activity and failures over time.
3. **Latency trend:** p50/p95 with seconds/milliseconds. Idle periods are gaps. The legend's Last value is the last observation, not a current latency while idle.
4. **Review-workload gauge:** 0–100%, minimum 20 eligible requests in the selected period; orange at 80%. High means review demand, not good accuracy. The matching alert uses a five-minute window.
5. **Source-range gauge:** 0–100%, latest up to 100 valid requests, minimum 20; orange at 20%. This is a temperature range-exceedance proxy, not verified drift or accuracy.
6. **Settings bar chart:** actual request mix in the selected period. Counts use neutral colors; more is not automatically worse.
7. **Active-alert table:** pending/firing alerts, severity and owner; empty means none. Alert rules contain response instructions.

The separate [Review evidence dashboard](http://127.0.0.1:3001/d/aeris-feedback) has four panels: evidence counts, review-coverage gauge, agree/disagree/unsure bars, and satisfaction on a 1–5 gauge. Counts and ratings are all-time persisted cohort records. No fake thresholds imply that a few demonstration ratings prove quality. Large wording panels were removed; definitions, denominators, time windows and limits are in tooltips and linked pages. Gauges are reserved for bounded proportions/ratings; trends and comparisons use appropriate charts.

The prominent Prometheus link now opens [Targets](http://127.0.0.1:9091/targets), showing **aeris 1/1 UP**, endpoint `http://app:8001/metrics`. UP means Prometheus can scrape the service; `aeris_model_ready` separately checks whether a valid model can load. A working scrape alone does not prove a usable diagnosis.

No requests means no latency observation. No agree/disagree responses means no agreement denominator. No independently verified labels means live accuracy is unmeasured. These are distinct evidence states, not zeros and not healthy green results.

The timeline uses explicit value mappings, consistent with [Grafana's state-timeline documentation](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/state-timeline/).

## Executed verification

- 21 Linux tests passed, including new calibrated-fusion, uncertain-parent retention, and invalid-contract tests. The three new tests also passed on Windows.
- Five canonical CSV boundary checks and JavaScript syntax check passed.
- All six catalog models issued predictions; repeated request IDs returned the same record.
- All 7+4 dashboard panels' expressions succeeded for both test/genuine cohorts through Grafana's datasource proxy. Dashboard totals matched SQLite.
- 100 HTTP predictions: median **93.7 ms**, p95 **187.1 ms**, all below five seconds. Includes durable HTTP service; excludes browser rendering. Grafana histograms cover a narrower service timing and approximate buckets, so its p95 can differ.
- Input rejection, unsupported context, full-model explanation, feedback deduplication, backup restore, restart persistence, origin/host boundaries passed.
- Missing-model failure triggered the availability alert, blocked diagnosis, recovered, and retained feedback. Raw evidence records the actual alert lifecycle.
- Browser verified gauge scales, UP labels, request bars, review evidence, model/data evidence deep link and Prometheus target UP. Screenshots accompany this report.

## Deployment and reproducibility

Project: `C:\MSAIS\EGN6216-AI SYSTEMS\AHU_PROJECT\aeris-data-warmup_updated`. Branch `aeris-operator-model-20261117`; prior dirty work and history preserved, no commit/push performed. Data and model binaries remain local; the evidence archive excludes raw data, weights, SQLite and environment directories.

`docker compose up -d --build` runs the loopback-only release. AERIS: http://127.0.0.1:8510 ; API: http://127.0.0.1:8011 ; Grafana: http://127.0.0.1:3001 ; Prometheus: http://127.0.0.1:9091 . The legacy Streamlit UI remains at port 8511. Do not delete the feedback volume.

Current scripts: 28 boost screen, 29 dashboard generator, 30 constrained fusion, 31 extra-sensor study, 32 package/evaluate, 33 operational verification, 34 assurance, 35 API verification, 36 failure/recovery. The isolated Windows experiment interpreter is `C:\Users\mhars\Documents\Codex\2026-11-16\files-pasted-by-the-user-implement\work\nov17b\env\Scripts\python.exe`; CatBoost is pinned in requirements. Archived older verification scripts intentionally describe prior dashboard releases and should not be used as current panel-count checks.

Rollback model selection/catalog are saved as `experiments/operator_redesign_20261117/prior_deployment_selection.json` and `prior_release_catalog.json`. Copy them back to `data/deployment_selection.json` and `data/release_catalog.json` to select the previous models on the current compatible image. Readiness reloads the selection. The prior `aeris-local:20261117` image remains available; current CatBoost bundles require the new image. Do not change model artifacts without regenerating integrity manifests.

For the professor: present a reproducible engineering and research project with measured gains, calibrated confidence, explicit scope, tested failure containment, explainability checks, persistent feedback and operational monitoring. Do not claim 90–100% universal accuracy, proven adversarial defense, successful cross-building transfer, or production certification. Independent AHU conditions, technician evaluation and field security/operational validation remain outstanding.
