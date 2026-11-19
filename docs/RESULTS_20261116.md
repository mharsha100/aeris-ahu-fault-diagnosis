# AERIS executed results — November 16, 2026

The local Docker demo deploys **e2e_anchor_v2_s42**, selected on source development evidence. It is research decision support, not field validation. Every deployed prediction combines RF and TNN. No target score selected hyperparameters, thresholds, or a winner. All reported target results are retrospective. Browser visual verification is pending; API and Streamlit source-level tests are separate evidence.

## Checkout, provenance and scope

Work is in `C:\MSAIS\EGN6216-AI SYSTEMS\AHU_PROJECT\aeris-data-warmup_updated`, branch `aeris-end-to-end-20261116`, starting at `961876af8df55a203f3305cfc58a453ff727fec7` from `milestone4-final-20261104-194011`. The starting checkout was clean. Changes are uncommitted and have not been pushed. `git_status_final.txt` and `changed_files.txt` capture the final working tree; historical milestone evidence is preserved.

All 47 selected raw inputs were present and hash-matched. Source has 453,645 rows; target has 37,439. No duplicate row IDs, duplicate recording timestamps, nonincreasing timestamps, or non-minute intervals were found. Source has no missing core values; target has 91 missing temperature values. The 470 channel conversion checks passed, and the fitted scaler exactly matched 140,402 eligible source-training rows. Existing prepared windows and frozen splits were reused. There are 27,720 training, 6,930 validation, 6,930 calibration and 6,929 later-source-test windows. Target adaptation/feedback/test have 1,420/1,136/1,136 unlabeled prepared windows.

Source is G36-1wk; targets are SZVAV and SZCAV. No authoritative AHU1/AHU2 numeric alias was found. Blocks do not overlap across partitions, but they reuse recordings and only three shared-weather source conditions. Thousands of windows are not thousands of independent cases. The five-independent-case floor is unmet. No confidence interval based on treating windows as independent is reported. Later-source scores are dependent blocked evaluation, not unseen-condition generalization.

October complete bundles reproduced source-validation macro F1 exactly: V0 0.688665 and V1-attention 0.688413. M4 reports were verified at approximately 0.683975/0.684224/0.686507 source and 0.194156/0.201800/0.181100 retrospective target for V0/V1/V2. Complete M4 forests are absent locally, so those M4 numbers are historical reports, not a fresh exact reproduction. The available download had temporal weights only. The historical V2 adaptation gate selected two windows from one day and one class and skipped an update.

## Executed tuning and model choice

The search executed 19 temporal trials: one two-epoch pilot, three fresh 50-epoch architecture anchors, six ten-epoch screens and nine 50-epoch finalists across seeds 42/43/44. Total completed temporal epochs: 662. Pilot runtime was 30.4 seconds. The declared runtime budget was 14,400 seconds. Each epoch and completed trial was saved. A cross-device serialization check failed once; saved artifacts were recovered, and serialization was then checked on the same CPU. Separate CPU/CUDA numerical differences and any class/review changes are recorded in `deployment_precision_audit.csv`. Recovered screening wall time is incomplete; do not add its epoch-only time to claim an exact total search duration.

RF varied trees 100/200/300, depth 12/18/24, leaf size 1/3/6, feature subsampling 0.5/0.7/sqrt and balanced/no class weighting. TNN varied width 16/32/48, dropout .05/.15/.30, learning rate .0005/.001/.002, decay .00001/.0001/.001, batch 128/256/512 and weighted/unweighted loss. Both branches remained in fusion weights .25/.50/.75/.90 RF. These were coupled reproducible configurations, not isolated causal effects of each parameter. V0 final state, V1 attention and V2 combined representations were trained fresh. The 29-step theoretical receptive field already spans the 15-sample input, so increasing it was not justified. Existing RF features already include rolling mean/std/min/max/last/trend; no unsupported missing-value imputation was added. The short small-width configuration did not meet diagnostic guards and was not promoted for speed.

The checkpoint rule was best source-development macro F1 with loss tie-break, no early stop. Calibration used only reserved source calibration. Target labels were opened only after selection was written. Predeclared promotion guards require every finalist seed to avoid increased fault-to-normal and accepted false-normal counts, with no class recall loss over .03; mean F1 must beat its matched anchor. Passing families rank by mean F1, then fewer false-normal errors, with fixed seed 42 as representative. Thresholds remain provisional. Rejected families retain their anchor.

| version | anchor_F1 | finalist_mean | finalist_std | finalist_worst | retained | selected | reason |
|---|---|---|---|---|---|---|---|
| v0 | 0.68890 | 0.69631 | 0.00120 | 0.69467 | False | e2e_anchor_v0_s42 | e2e_final_v0_s43 failed false-normal/recall guard |
| v1 | 0.68946 | 0.69734 | 0.00203 | 0.69449 | False | e2e_anchor_v1_s42 | e2e_final_v1_s43 failed false-normal/recall guard; e2e_final_v1_s44 failed false-normal/recall guard |
| v2 | 0.68997 | 0.69457 | 0.00257 | 0.69095 | False | e2e_anchor_v2_s42 | e2e_final_v2_s43 failed false-normal/recall guard; e2e_final_v2_s44 failed false-normal/recall guard |

Full-budget results below use the actual CPU deployment path; the complete 19-trial table preserves CUDA training results, parameters and hashes.

| name | seed | macro_f1 | fault_to_normal | normal_false_alarm | seconds |
|---|---|---|---|---|---|
| e2e_anchor_v0_s42 | 42 | 0.68890 | 596 | 255 | 184.13268 |
| e2e_final_v0_s42 | 42 | 0.69749 | 573 | 239 | 245.24637 |
| e2e_final_v0_s43 | 43 | 0.69467 | 749 | 204 | 210.73839 |
| e2e_final_v0_s44 | 44 | 0.69678 | 534 | 255 | 207.88155 |
| e2e_anchor_v1_s42 | 42 | 0.68946 | 601 | 253 | 190.55509 |
| e2e_final_v1_s42 | 42 | 0.69847 | 559 | 238 | 235.15361 |
| e2e_final_v1_s43 | 43 | 0.69449 | 715 | 211 | 255.10017 |
| e2e_final_v1_s44 | 44 | 0.69905 | 759 | 193 | 218.25703 |
| e2e_anchor_v2_s42 | 42 | 0.68997 | 635 | 245 | 171.56877 |
| e2e_final_v2_s42 | 42 | 0.69613 | 521 | 256 | 144.27637 |
| e2e_final_v2_s43 | 43 | 0.69095 | 483 | 270 | 152.40372 |
| e2e_final_v2_s44 | 44 | 0.69663 | 723 | 208 | 131.94482 |

Selected source-development macro F1 is **0.689975**, with **635/6,468** fault windows classified normal and **245/462** normal windows assigned a fault. Training macro F1 is 0.803092; the training/development gap remains substantial. The V2 anchor has more pre-review false-normal windows than the V0/V1 anchors (635 versus 596/601); its selection is the declared macro-F1 ranking among retained anchors, not a claim that it wins every safety metric. At the fixed .90 gate, 3416/6,930 windows require review and 0/3514 accepted fault windows are classified normal. In this development set all normal windows also require review. Review-or-fault-escalation recall is 1.0000; this window statistic does not establish reliable technician outcomes or independent-case performance. Source classification is not evidence of target reliability.

Matched branches:

| model | branch | macro_f1 | brier | log_loss |
|---|---|---|---|---|
| e2e_anchor_v0_s42 | hybrid | 0.68890 | 0.28411 | 0.48215 |
| e2e_anchor_v0_s42 | RF-only | 0.68843 | 0.28969 | 0.49474 |
| e2e_anchor_v0_s42 | TNN-only | 0.66737 | 0.29239 | 0.49329 |
| e2e_anchor_v1_s42 | hybrid | 0.68946 | 0.28464 | 0.48312 |
| e2e_anchor_v1_s42 | RF-only | 0.68843 | 0.28969 | 0.49474 |
| e2e_anchor_v1_s42 | TNN-only | 0.66998 | 0.29657 | 0.50286 |
| e2e_anchor_v2_s42 | hybrid | 0.68997 | 0.27903 | 0.47383 |
| e2e_anchor_v2_s42 | RF-only | 0.68843 | 0.28969 | 0.49474 |
| e2e_anchor_v2_s42 | TNN-only | 0.66896 | 0.28603 | 0.48353 |

An isolated RF class-weight experiment held the network, RF size/depth, seed and fusion fixed. Balanced weighting gave F1 .688902, 596 fault-to-normal and 255 normal false alarms; removing weighting gave .658811, 200 and 388. Reduced missed faults came with markedly worse normal recognition and macro F1, so balancing was retained. Per-class results are in `weighting_ablation.json`.

Temperature scaling versus raw fusion:

| bundle | method | macro_f1 | brier | log_loss |
|---|---|---|---|---|
| e2e_anchor_v0_s42 | raw weighted branches | 0.68885 | 0.28596 | 0.47924 |
| e2e_anchor_v0_s42 | source-calibration temperature scaling | 0.68890 | 0.28411 | 0.48215 |
| e2e_anchor_v1_s42 | raw weighted branches | 0.68959 | 0.28633 | 0.47975 |
| e2e_anchor_v1_s42 | source-calibration temperature scaling | 0.68946 | 0.28464 | 0.48312 |
| e2e_anchor_v2_s42 | raw weighted branches | 0.69064 | 0.28502 | 0.47905 |
| e2e_anchor_v2_s42 | source-calibration temperature scaling | 0.68997 | 0.27903 | 0.47383 |

Reliability bins, proper probability scores and the .50–.99 review sweep are saved. Calibration does not guarantee increased F1 or target calibration. The .90 threshold was retained rather than chosen using retrospective target results.

Post-selection descriptive evaluation:

| bundle | role | windows | macro_f1 | fault_to_normal | mandatory_review_fraction | scope |
|---|---|---|---|---|---|---|
| e2e_anchor_v0_s42 | source_test | 6929 | 0.63237 | 382 | — | later source day within same conditions; not independent generalization |
| e2e_anchor_v0_s42 | target_test | 1136 | 0.22736 | 0 | 1.00000 | retrospective target; never used for selection |
| e2e_anchor_v1_s42 | source_test | 6929 | 0.63301 | 386 | — | later source day within same conditions; not independent generalization |
| e2e_anchor_v1_s42 | target_test | 1136 | 0.22921 | 0 | 1.00000 | retrospective target; never used for selection |
| e2e_anchor_v2_s42 | source_test | 6929 | 0.65418 | 432 | — | later source day within same conditions; not independent generalization |
| e2e_anchor_v2_s42 | target_test | 1136 | 0.22432 | 0 | 1.00000 | retrospective target; never used for selection |

The historical target macro-F1 goal .70 is not a validated achievement. Target inputs always require review. Source-only parents remain deployed: cross-setting updates are visibly skipped for unsupported applicability, and genuine-feedback updates are skipped because no independently verified, aligned eligible labels exist. Per-window gate rejections and candidate diversity are saved; gates were not loosened to force an update. The explicit reviewed feedback-candidate command enforces genuine provenance, reserved feedback data, case-separated partitions, per-class case floors and parent comparison. This successful-update path is implemented but unexercised with real eligible labels.

## Operational groups, robustness and explanations

| group | class_name | windows | recordings | recall | fault_to_normal_rate | review_rate | evidence |
|---|---|---|---|---|---|---|---|
| CoolingSeason | normal | 154 | 1 | 0.55195 | 0.03340 | 0.50909 | insufficient independent cases (<5) |
| CoolingSeason | damper_fault | 1078 | 7 | 0.84601 | 0.03340 | 0.50909 | insufficient independent cases (<5) |
| CoolingSeason | cooling_valve_fault | 1078 | 7 | 0.77087 | 0.03340 | 0.50909 | insufficient independent cases (<5) |
| CoolingSeason | heating_valve_fault | 0 | 0 | — | 0.03340 | 0.50909 | insufficient independent cases (<5) |
| HeatingSeason | normal | 154 | 1 | 0.11039 | 0.03154 | 0.45671 | insufficient independent cases (<5) |
| HeatingSeason | damper_fault | 1078 | 7 | 0.85529 | 0.03154 | 0.45671 | insufficient independent cases (<5) |
| HeatingSeason | cooling_valve_fault | 0 | 0 | — | 0.03154 | 0.45671 | insufficient independent cases (<5) |
| HeatingSeason | heating_valve_fault | 1078 | 7 | 0.81076 | 0.03154 | 0.45671 | insufficient independent cases (<5) |
| ShoulderSeason | normal | 154 | 1 | 0.74675 | 0.22959 | 0.51299 | insufficient independent cases (<5) |
| ShoulderSeason | damper_fault | 1078 | 7 | 0.72727 | 0.22959 | 0.51299 | insufficient independent cases (<5) |
| ShoulderSeason | cooling_valve_fault | 1078 | 7 | 0.68646 | 0.22959 | 0.51299 | insufficient independent cases (<5) |
| ShoulderSeason | heating_valve_fault | 0 | 0 | — | 0.22959 | 0.51299 | insufficient independent cases (<5) |

Every source group is marked insufficient independent evidence. Empty class/season combinations are unsupported, not zero-accuracy measurements. Target setting/class counts appear in `subgroups.csv`, with 100% mandatory review. An explicit pandas/sklearn group analysis is used as the MetricFrame equivalent because the needed counts and denominators are directly inspectable.

The bounded test uses the same 100 source windows (25 per class, 75 fault windows) for each bundle and six scenarios. Both branches are regenerated from the same modified raw sensor input. Gaussian temperature noise is sigma .2°C clipped to ±.6°C; the supply offset is +1°C; dropout introduces one missing sample; stuck supply holds the first value. No claim of worst-case adversarial robustness is made. F1 is conditional on an actual prediction; all-blocked cases have no F1. The case table also records every status and input ID.

| bundle | case | windows | fault_windows | predictions | blocked | review_required | macro_f1 | false_normal | accepted_false_normal | delta_macro_f1_vs_clean |
|---|---|---|---|---|---|---|---|---|---|---|
| e2e_anchor_v2_s42 | clean | 100 | 75 | 100 | 0 | 56 | 0.74887 | 7 | 0 | 0.00000 |
| e2e_anchor_v2_s42 | dropout | 100 | 75 | 0 | 100 | 0 | — | 0 | 0 | — |
| e2e_anchor_v2_s42 | noise_0.2C | 100 | 75 | 100 | 0 | 61 | 0.71447 | 1 | 0 | -0.03440 |
| e2e_anchor_v2_s42 | offset_supply_1C | 100 | 75 | 100 | 0 | 59 | 0.53649 | 1 | 0 | -0.21238 |
| e2e_anchor_v2_s42 | stuck_supply | 100 | 75 | 100 | 0 | 58 | 0.60129 | 2 | 0 | -0.14758 |
| e2e_anchor_v2_s42 | unknown_setting | 100 | 75 | 100 | 0 | 100 | 0.74887 | 7 | 0 | 0.00000 |
| v0 | clean | 100 | 75 | 100 | 0 | 55 | 0.74847 | 7 | 0 | 0.00000 |
| v0 | dropout | 100 | 75 | 0 | 100 | 0 | — | 0 | 0 | — |
| v0 | noise_0.2C | 100 | 75 | 100 | 0 | 57 | 0.67740 | 2 | 0 | -0.07107 |
| v0 | offset_supply_1C | 100 | 75 | 100 | 0 | 57 | 0.52632 | 0 | 0 | -0.22215 |
| v0 | stuck_supply | 100 | 75 | 100 | 0 | 56 | 0.59608 | 2 | 0 | -0.15239 |
| v0 | unknown_setting | 100 | 75 | 100 | 0 | 100 | 0.74847 | 7 | 0 | 0.00000 |
| v1_attention | clean | 100 | 75 | 100 | 0 | 51 | 0.78087 | 10 | 0 | 0.00000 |
| v1_attention | dropout | 100 | 75 | 0 | 100 | 0 | — | 0 | 0 | — |
| v1_attention | noise_0.2C | 100 | 75 | 100 | 0 | 51 | 0.74653 | 8 | 0 | -0.03434 |
| v1_attention | offset_supply_1C | 100 | 75 | 100 | 0 | 53 | 0.57845 | 3 | 0 | -0.20242 |
| v1_attention | stuck_supply | 100 | 75 | 100 | 0 | 52 | 0.68562 | 10 | 0 | -0.09526 |
| v1_attention | unknown_setting | 100 | 75 | 100 | 0 | 100 | 0.78087 | 10 | 0 | 0.00000 |

The selected model has 0 high-confidence development errors, including 0 high-confidence false-normal errors before application review. The 20 highest-priority errors are saved with recording IDs and disagreement. Explanation tests covered 6 representative windows spanning the classes; exact-repeat consistency passed for 6. Largest grouped-effect change after +.05°C was 0.03940; mean explanation runtime was 0.120 seconds. Full-hybrid group replacement is sensitivity to a training-mean reference, not a physical cause; off-manifold and correlation limits are displayed. Independent technician usefulness remains untested.

## Implemented controls and deployment evidence

NIST Govern/Map/Measure/Manage and the actual M2 risk matrix are mapped in `docs/TRUSTWORTHINESS.md`. Controls are implemented in the application and workflows: strict shape/unit/range/fan/occupancy/timestamp checks; mandatory unsupported-setting review; confidence/agreement/source-bound reasons; integrity-checked bundles; durable idempotent predictions; versioned feedback history; explicit training/promotion; matched selection guards; bounded-label metrics; and tested failure recovery. Original R1–R8 remain visible, with R9 explanation risk added. Field false-normal risk remains Critical; a software test does not reduce its physical consequence.

The app, Prometheus and Grafana use loopback ports 8510, 9091 and 3001; the API is 8011. Docker uses a pinned Python 3.11 CPU image; CUDA training was verified on the RTX 4070 Laptop GPU with PyTorch 2.9.1+cu128. Container GPU passthrough is unverified. A one-epoch V2 CPU container training/reload smoke test completed and is explicitly excluded from model selection. Data and fitted artifacts are mounted, not baked into the image. SQLite, Prometheus and Grafana persist in named volumes.

The final live API test passed valid inference, malformed rejection, physical-bound edge rejection, unknown-setting review, stable prediction IDs, duplicate-feedback protection and full-hybrid explanation. All 15 regression tests passed on Windows and in the Linux container. Of 100 warm HTTP requests, 100 completed under five seconds; p50 0.106s, p95 0.220s. These exclude browser rendering. The separate warm Streamlit AppTest run used 100 requests: p50 0.314s, p95 0.635s, 100/100 under five seconds; cold first diagnosis took 2.751s. It is server widget execution, not browser-paint evidence. The original 95/100 rendered-UI-under-five-seconds target remains unverified.

| bundle | cpu_p50_seconds | cpu_p95_seconds | samples | scope | bytes |
|---|---|---|---|---|---|
| e2e_anchor_v0_s42 | 0.07890 | 0.15588 | 100 | warm hybrid inference only; same CPU process; excludes UI | 12251752 |
| e2e_anchor_v1_s42 | 0.07369 | 0.13496 | 100 | warm hybrid inference only; same CPU process; excludes UI | 12256958 |
| e2e_anchor_v2_s42 | 0.09429 | 0.14223 | 100 | warm hybrid inference only; same CPU process; excludes UI | 12265844 |

Actual controlled missing-model alert fired: **True**; recovery: **True**; feedback unchanged across restart: **True**. A cache bug discovered in recovery testing was fixed: a restored bundle must reload when the cached model is absent even if its identity is unchanged. The regression test and actual successful inference before restart both passed. Grafana's provisioned 24-panel dashboard and datasource returned real Prometheus series. Current genuine predictions/feedback/verified labels are 0/0/0; test predictions/feedback are 413/5. No test opinion is presented as independent ground truth. Missing genuine agreement/satisfaction return no data, not artificial zero quality.

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
