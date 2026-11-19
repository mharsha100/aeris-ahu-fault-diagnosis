# AERIS improvement and readiness report — November 17, 2026

## Outcome

The restored application and monitoring stack are running locally. The latest search completed **nine fresh hybrid models × 50 epochs = 450 epochs**, seven classical diagnostic fits, and nine source-calibration decision fits. The candidate family uses three newly fitted physical-feature RFs (one per seed, shared across pooling comparisons) and nine freshly initialized temporal models. These are additional to November 16's documented 19 temporal runs / 662 epochs. Each architecture still combines Random Forest and its own temporal network. The proposed physical representations did **not** satisfy all preregistered release guards across every seed, so the verified parent models remain deployed. The requested high-accuracy, production-validated model has **not been achieved**. Present this as a working, supervised research system with measured limitations.

The ~69% figure is **macro F1**, not accuracy. Retained V2 source-development accuracy is 76.19%, macro F1 0.6900, macro precision 69.58%, and macro recall 70.47%. Confidence is calibrated, not forced upward. Unsupported targets always require review.

## What changed and why

1. **Model investigation:** added 46 deterministic physical interactions to the RF's 66 summaries (112 total), including mixing residuals and temperature/command relationships. Added a common physical-context residual head to each distinct temporal architecture. Trained fresh V0 last-state, V1 attention and V2 combined models for 50 completed epochs at seeds 42/43/44. Tested RF, ExtraTrees and histogram gradient boosting as diagnostic comparators. Classical-only diagnostics do not replace the required hybrid.
2. **Controlled calibration:** fitted fusion and class-decision factors on source calibration only. Reviewed macro F1, every class recall, all false-normal predictions and accepted false-normal predictions. Every seed must gain at least 0.01 macro F1 without more missed faults or a recall drop over 0.03. Rejected results remain available; no target-score tuning or relabeling occurred.
3. **Interface:** restored the earlier forest-green/cream design and connected it to the real Python hybrid service. Added raw signal plots, architecture selection, all four probabilities, branch comparison, uncertainty reasons, exact grouped explanations, linked review forms, input CSV validation, and audit downloads. Selections clear stale results; pending requests lock their controls. Model evidence, assurance, monitoring and semester history have dedicated views.
4. **Explainability:** replaced simple group removal with exact four-group Shapley values, evaluating all 16 coalitions through the complete RF + temporal model. The view reports baseline, predicted probability, contributions and additivity residual. Binary fan/occupancy context stays fixed. This explains model behavior relative to a source-mean reference, not physical causation.
5. **Monitoring:** the Prometheus link opens a populated readiness graph, with separate targets/alerts links. Grafana has 26 provisioned panels, explicit health mappings, feedback support and cohort filters, and specific missing-value explanations. Counts come from actual persisted records. No artificial genuine usage was added.
6. **Operations:** kept durable SQLite IDs, feedback history and idempotency; added strict CSV parsing, host/origin and JSON-write boundaries, catalog-only version selection, real-browser checks, and a non-destructive online backup/restore rehearsal. Existing integrity, failure recovery and rollback controls remain active. The legacy Streamlit workspace is still available separately.

## Model measurements

All source-development comparisons use the same 6,930 windows. Later source-day and previously exposed target scores are descriptive, not independent site validation. Target results below were not used to select candidates.

| version | partition | n | accuracy | macro_F1 | macro_precision | macro_recall | ECE | Brier | review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | Source development | 6930 | 0.7631 | 0.6889 | 0.6992 | 0.6987 | 0.0238 | 0.2841 | 0.5078 |
| V0 | Later source day | 6929 | 0.7176 | 0.6324 | 0.6705 | 0.6167 | 0.0395 | 0.3484 | None |
| V0 | Retrospective target | 1136 | 0.3442 | 0.2274 | 0.1994 | 0.3442 | 0.2583 | 0.8863 | 1.0000 |
| V1 | Source development | 6930 | 0.7631 | 0.6895 | 0.6959 | 0.7013 | 0.0260 | 0.2846 | 0.5095 |
| V1 | Later source day | 6929 | 0.7177 | 0.6330 | 0.6678 | 0.6189 | 0.0375 | 0.3488 | None |
| V1 | Retrospective target | 1136 | 0.3468 | 0.2292 | 0.2024 | 0.3468 | 0.2916 | 0.8832 | 1.0000 |
| V2 | Source development | 6930 | 0.7619 | 0.6900 | 0.6958 | 0.7047 | 0.0307 | 0.2790 | 0.4929 |
| V2 | Later source day | 6929 | 0.7317 | 0.6542 | 0.6723 | 0.6469 | 0.0345 | 0.3319 | None |
| V2 | Retrospective target | 1136 | 0.3398 | 0.2243 | 0.1694 | 0.3398 | 0.2154 | 0.8974 | 1.0000 |

The following source-development counts distinguish raw classifier mistakes from decisions that pass the review gate:

| version | faults_predicted_normal | fault_support | normal_false_alarms | normal_support | review_rate | accepted_windows | accepted_accuracy | accepted_false_normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | 596 | 6468 | 255 | 462 | 0.5078 | 3411 | 1.0000 | 0 |
| V1 | 601 | 6468 | 253 | 462 | 0.5095 | 3399 | 1.0000 | 0 |
| V2 | 635 | 6468 | 245 | 462 | 0.4929 | 3514 | 1.0000 | 0 |

### Precision and recall

| version | condition | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- | --- |
| V0 | normal | 0.2578 | 0.4481 | 0.3273 | 462.0000 |
| V0 | damper_fault | 0.7826 | 0.8281 | 0.8047 | 3234.0000 |
| V0 | cooling_valve_fault | 0.9223 | 0.7106 | 0.8027 | 2156.0000 |
| V0 | heating_valve_fault | 0.8343 | 0.8080 | 0.8209 | 1078.0000 |
| V1 | normal | 0.2580 | 0.4524 | 0.3286 | 462.0000 |
| V1 | damper_fault | 0.7922 | 0.8169 | 0.8044 | 3234.0000 |
| V1 | cooling_valve_fault | 0.9009 | 0.7250 | 0.8034 | 2156.0000 |
| V1 | heating_valve_fault | 0.8324 | 0.8108 | 0.8214 | 1078.0000 |
| V2 | normal | 0.2547 | 0.4697 | 0.3303 | 462.0000 |
| V2 | damper_fault | 0.7984 | 0.8095 | 0.8039 | 3234.0000 |
| V2 | cooling_valve_fault | 0.8993 | 0.7287 | 0.8050 | 2156.0000 |
| V2 | heating_valve_fault | 0.8308 | 0.8108 | 0.8207 | 1078.0000 |

Macro F1 averages the four class F1 scores equally. Accuracy weights every window equally. Macro recall is also balanced accuracy. ECE is ten-bin expected calibration error; lower is better, but binning and sample dependence limit its interpretation. Multiclass Brier is summed squared probability error (range 0–2); lower is better. Log loss for retained V2 is 0.4738. None of these metrics proves transfer safety.

### Improvement candidates

| family | version | mean_F1 | seed_std | worst_F1 | promoted |
| --- | --- | --- | --- | --- | --- |
| physics | V0 | 0.6950 | 0.0021 | 0.6933 | False |
| calibrated_physics | V0 | 0.6860 | 0.0040 | 0.6820 | False |
| physics | V1 | 0.6974 | 0.0008 | 0.6963 | False |
| calibrated_physics | V1 | 0.6863 | 0.0058 | 0.6786 | False |
| physics | V2 | 0.6954 | 0.0006 | 0.6948 | False |
| calibrated_physics | V2 | 0.6836 | 0.0015 | 0.6815 | False |

| candidate | macro_F1 | fault_to_normal |
| --- | --- | --- |
| rf_physics_bal | 0.6902 | 269 |
| rf_physics_none | 0.6854 | 346 |
| extra_physics | 0.6934 | 344 |
| hgb_base_15 | 0.6805 | 857 |
| hgb_base_31 | 0.6923 | 547 |
| hgb_physics_15 | 0.6904 | 932 |
| hgb_physics_31 | 0.6934 | 708 |

The full per-seed rejection reasons are in `release_trials.csv` and `selection.json`. Seed standard deviations describe training variability, not independent-building confidence intervals. Full curves, chosen checkpoints, fusion sweeps, calibration factors, plans and hashes are retained. No epoch was silently skipped; best development checkpoints were selected after all 50 epochs.

## Risk management, robustness and trustworthiness

We use the original M2 risk matrix and [NIST AI RMF Govern, Map, Measure, Manage](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) to organize controls. NIST is a process framework, not a numeric certification score. Implementation uses NumPy/pandas/scikit-learn/PyTorch for reproducible measurement, application validation and review rules, SQLite for traceability, integrity hashes and guarded selection for model changes, and Prometheus/Grafana for operational detection. No risk is declared eliminated merely because a test passed.

| control | measure | result | limit |
| --- | --- | --- | --- |
| Bounded attack: ±0.2°C / ±0.01 commands | Accepted wrong / false normal | 0/40; 0/30 faults | Raw attacked accuracy 60.0%; review 87.5%; 64 random attacks/window, not exhaustive |
| Bounded attack: ±0.5°C / ±0.02 commands | Accepted wrong / false normal | 0/40; 0/30 faults | Raw attacked accuracy 57.5%; review 95.0%; 64 random attacks/window, not exhaustive |
| Bounded attack: ±1.0°C / ±0.05 commands | Accepted wrong / false normal | 1/40; 0/30 faults | Raw attacked accuracy 57.5%; review 97.5%; 64 random attacks/window, not exhaustive |
| Exact grouped Shapley | Maximum additivity residual | 1.11e-16 | 20 windows, four sensor groups; not causal or human-usefulness validation |
| Explanation repeatability | Identical repeated explanations | 100.0% | Same model, input and reference; deterministic repeatability only |
| Explanation stability | Top absolute group after +0.05°C | 95.0% | Same explained class; 20 windows, 5 per class |
| Scope protection | Mandatory target review | 100% | Protection through abstention; target diagnostic performance remains weak |
| Model release governance | Every-seed F1 / recall / missed-fault guards | 0/6 candidate families passed | Selection is source development; validation reuse can overfit |

### Adversarial test details

| temperature_budget_C | command_budget | windows | fault_windows | clean_accuracy | attack_accuracy | clean_correct_windows | successful_attacks | attack_success_rate | windows_with_any_accepted_wrong_class | windows_with_any_accepted_false_normal | worst_case_review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 0.0100 | 40 | 30 | 0.8500 | 0.6000 | 34 | 10 | 0.2941 | 0 | 0 | 0.8750 |
| 0.5000 | 0.0200 | 40 | 30 | 0.8500 | 0.5750 | 34 | 11 | 0.3235 | 0 | 0 | 0.9500 |
| 1.0000 | 0.0500 | 40 | 30 | 0.8500 | 0.5750 | 34 | 11 | 0.3235 | 1 | 0 | 0.9750 |

This defensive black-box experiment used 40 source-development windows (10/class), 64 bounded randomized perturbations plus the clean input per window/budget, and regenerated both branches from the same altered raw window. The fault subset has 30 windows. Attack success is conditional on initially correct predictions. Raw robust accuracy counts any discovered misclassification as failure. Accepted-error counts examine all tested candidates under the complete review policy; the displayed worst-input review rate uses the candidate minimizing the true-class probability. Strong review rates represent abstention cost, not robust classification accuracy. These are small, dependent development samples and a finite search, not an adversarial certificate or guarantee against future attacks.

At the source-development review gate, V2 reviews 49.29% of windows, accepts 3514, and has 0 accepted false-normal predictions in this sample. Accepted accuracy is 100.00%. **All 462 normal-reference windows are reviewed and there are 0 accepted normal predictions.** The high accepted score therefore does not demonstrate reliable autonomous four-class diagnosis or safe equipment clearance.

### Explanation measurements

Twenty windows (five/class), fixed model and source-mean reference: maximum additivity error **1.11e-16**, identical repeatability **100.0%**, same-class top-group stability under a +0.05°C change **95.0%**, mean runtime **35.1 ms**. Exact numerical additivity is expected from exact Shapley enumeration; it is not 100% semantic correctness. Off-manifold reference combinations and correlated sensors remain limitations. Independent technician usefulness has not been measured.

### Why strong data did not automatically yield a high score

The cited [Scientific Data dataset paper](https://www.nature.com/articles/s41597-025-06179-y) describes labeled AHU datasets. It does not guarantee separability for this ten-channel, four-class, blocked-window experiment. Here normal has only 462/6,930 development windows; fault severity and operating conditions affect observability; source and target AHUs differ; and a scenario fault label can persist while its actuator is inactive. The new physical interactions and model comparisons tested plausible improvements without changing the evaluation task.

The observability audit found no exact full-precision duplicate windows with conflicting labels. Quantizing measurements creates some cross-label collisions; that is an illustrative resolution study, **not** proof of a hard ceiling on the original model. More independent operating/fault conditions, sensor-semantic verification and appropriately designed new features are needed before claiming high field accuracy. Randomly mixing neighboring windows or tuning against target labels would make the score less credible.

## Verification and deployment

The main UI is http://127.0.0.1:8510/ ; legacy Streamlit http://127.0.0.1:8511/ ; Grafana http://127.0.0.1:3001/d/aeris-operations ; Prometheus http://127.0.0.1:9091/ . All host ports bind to loopback. This is a local deployment, not cloud production.

Python regression checks: 18 passed. Browser CSV contract checks: 5 passed. All three catalog versions returned valid hybrid predictions. A real browser submitted one demonstration review twice: the ledger contains exactly one prediction, one current feedback record, and one history entry for that action. Restart preserved all summary totals. Backup restore reproduced every table's row count and content hash in an independent temporary database. Grafana queried actual Prometheus values matching SQLite totals; 26 panels and healthy scrape targets were verified.

Operational checks: cross_origin, text_form, untrusted_host, path_traversal, negative_index, all_catalog_versions_and_idempotency, backup_restore, restart_preserves_all_feedback_totals, grafana_26_panels, grafana_live_totals_match_sqlite, test_data_not_presented_as_genuine. The current release also executed a missing-default-bundle failure: its inference request returned an error without a diagnosis, `AerisUnavailable` fired, restoring the selection cleared the alert and recovered inference, and feedback survived another restart. The 100-window browser test also triggered the review-burden warning; this is an observed workload signal, not an availability failure.

Browser measurements and visual checks:

```json
{
  "date": "2026-11-17T06:46:20.082Z",
  "browser": "Actual in-app browser; default viewport",
  "checks": {
    "rendered_hybrid_result": true,
    "feedback_form_saved_and_duplicate_save_exercised": true,
    "exact_grouped_explanation_visible": true,
    "selection_clears_stale_result": true,
    "invalid_csv_error_visible": true,
    "target_requires_review": true,
    "model_evidence_and_per_class_metrics_visible": true,
    "assurance_limits_visible": true,
    "semester_evidence_visible": true,
    "prometheus_graph_real_series_visible": true,
    "grafana_health_and_missing_value_explanations_visible": true
  },
  "latency": {
    "requests": 100,
    "unique_prediction_ids": 100,
    "p50_milliseconds": 235.10000000149012,
    "p95_milliseconds": 322.67999999746667,
    "max_milliseconds": 463.69999999552965,
    "under_5000_milliseconds": 100,
    "scope": "Warm V2 source-development windows 0\u00e2\u20ac\u201c99. Browser window-fetch + inference + two animation frames. Excludes startup, model switching and concurrent-user load; local CPU evaluation was also running."
  },
  "proof_files": [
    "diagnosis.jpg",
    "explanation.jpg",
    "monitoring.jpg",
    "prometheus.jpg",
    "grafana.jpg"
  ]
}
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
