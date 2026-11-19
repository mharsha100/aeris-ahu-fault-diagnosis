# AERIS trustworthiness and risk register update

This November 16 addendum maintains the M2 risk identities and original ratings.
It applies [NIST AI RMF Govern, Map, Measure and Manage](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
to development and deployment. It does not approve field use. The owner roles
below are project responsibilities proposed for this course prototype; no
external organization has accepted an operational role.

| Function | Development and validation | Deployment and monitoring |
|---|---|---|
| Govern | Preserve provenance, split and five-independent-case floor; freeze selection criteria; version trials | Project operator owns startup/rollback; model reviewer owns evidence and promotion; no automatic training or actuation |
| Map | Record simulated G36 and real SZVAV/SZCAV scope, units and class/season gaps | Distinguish recorded-source input from unverified uploads; unsupported settings require review |
| Measure | Matched branches, class recall, false-normal errors, calibration, grouped metrics, bounded sensor tests | Durable requests, UI/model readiness, latency, source-scale shift, feedback denominators and group coverage |
| Manage | Reject unsafe finalists; preserve skipped adaptation and unavailable independent labels | Stop recommendations when bundle fails; alert and restore verified bundle; investigate sustained shift before changes |

The operational-group analysis uses explicit pandas groupby counts and the
same sklearn class metrics as a transparent equivalent of MetricFrame. No
demographic fairness claim is made. Recorded days and severity variants do
not create independent fault cases. Every source subgroup remains below the
five-independent-case reporting floor. Unsupported target settings remain
review-only. A sustained group gap requires new independent recordings and
retesting, not a threshold change chosen against retrospective target scores.

Robustness uses controlled raw-sensor perturbation rather than a classifier-only
adversarial library: both RF summaries and temporal inputs are regenerated from
the same changed 15-minute window. Tests include bounded 0.2°C noise clipped at
0.6°C, a +1°C supply sensor offset, dropouts, a stuck supply sensor and unknown
settings. These are defensive simulations, not a worst-case robustness proof.

Explanations replace complete sensor groups with fixed training-source means
and measure the full hybrid's response. The UI labels the counterfactual
reference and correlation limitations. Repeatability and runtime are tested;
independent technician usefulness has not been evaluated. No attribution is
presented as a physical cause.

| Likelihood / Impact | Acceptable | Tolerable | Unacceptable | Intolerable |
|---|---|---|---|---|
| Improbable | Low | Moderate | Moderate | High |
| Possible | Low | Moderate | High | Critical |
| Probable | Moderate | High | High | Critical |

| ID and failure scenario; affected party | Original assessment | Implemented control and evidence | Current residual and rationale | Owner; trigger and response |
|---|---|---|---|---|
| R1 ★ Sensor/fault mismatch makes transfer misleading; technician | Probable × Unacceptable = High | Raw hashes, Celsius/fraction contract, source/target scope, mandatory upload/target review; audit.json and service tests | Possible × Unacceptable = High. Dynamic equivalence and numeric AHU1/AHU2 alias remain unverified | Model reviewer; unmatched channels or modes → withhold transfer claims and seek site evidence |
| R2 Dependent windows inflate reported quality; course reviewer | Probable × Unacceptable = High | Frozen manifests, interval separation, source-only fitting, independent-case labels | Possible × Unacceptable = High. No interval overlap was found, but the same fault recordings span day blocks; independent generalization remains unsupported | Model reviewer; any shared minute across partitions → invalidate run; fewer than five cases → withhold claims |
| R3 ★ False-normal prediction delays fault inspection; technician/operations | Possible × Intolerable = Critical | Both branches, confidence/agreement/scope review, false-normal finalist guards, coverage sweeps; no equipment control | Possible × Intolerable = Critical for field reliance. Offline demonstration only; human review does not prove missed-fault risk is removed | Project operator; confirmed accepted false-normal → withdraw reassuring recommendation and require review |
| R4 Degraded/malformed sensor input changes diagnosis; technician | Probable × Unacceptable = High | Exact upload schema, finite/range/fan/timestamp checks, noise/offset/dropout/stuck tests | Possible × Unacceptable = High. Plausible offsets and stuck readings can evade bounds; measured degradation must remain visible | Operator; malformed input → reject; sustained shift → inspect sensor and units |
| R5 Rare fault/operating group is poorly served; technician | Probable × Unacceptable = High | Per-class recall, support, three-seed comparisons, group review rates | Possible × Unacceptable = High. Low independent support blocks class/season generalization | Model reviewer; low support/gap → acquire independent cases and retest mitigations |
| R6 Operating shift produces unsupported predictions; operations | Possible × Unacceptable = High | Fixed-source standardized temperature exceedance, confidence/review burden, setting coverage and mandatory target review | Possible × Unacceptable = High. Unlabeled signals cannot establish accuracy or causal drift | Operator; >20% source-scale exceedance with ≥20 recent requests for 30 s → inspect context and keep review |
| R7 ★ Bad feedback enters training; technician/model reviewer | Possible × Intolerable = Critical | Linked SQLite records, explicit test/genuine cohort, provenance, pending-alignment state, no UI training or automatic promotion | Improbable × Intolerable = High in this disabled-update demo. Genuine reviewed adaptation remains pending eligible labels | Model reviewer; unverified/test record enters training → block update and roll back; verify parent comparison before promotion |
| R8 Slow/unavailable service prevents review; operator | Possible × Tolerable = Moderate | CPU image, application/model readiness, persistent volumes, latency/failure/restart checks | Possible × Tolerable = Moderate. Single laptop remains a single point of failure | Operator; not ready for 15 s or p95 >5 s → inspect logs/bundle and restore parent |
| R9 Misleading explanation is treated as cause; technician | Possible × Unacceptable = High (new) | Full-hybrid grouped sensitivity, explicit reference and noncausal label, repeated-case checks | Possible × Unacceptable = High. Human usefulness and off-manifold effects unresolved | Model reviewer; unstable/confusing explanation → suppress it and retain raw decision audit |

Evidence is under experiments/end_to_end_20261116. A passing software test
does not lower the consequence of a missed physical fault. Current genuine
inspection support and eligibility must be read from /summary; simulated
reference feedback and test submissions cannot establish accuracy. The three
priority deployment signals are availability, fixed-source shift, and review
burden. Alert rules and response text are version controlled in deployment/alerts.yml.
