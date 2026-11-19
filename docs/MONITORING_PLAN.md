# AERIS monitoring plan

This is a short local deployment demonstration. It does not claim continuous
semester-long monitoring, prospective accuracy, or external alert delivery.
The Prometheus evaluation and scrape intervals are five seconds. Alerts are
visible in Prometheus; no external Alertmanager recipient is configured.

| Priority signal | Query and expected range | Trigger and support | Owner and response |
|---|---|---|---|
| Availability/readiness (R8) | `up{job="aeris"}`, `aeris_app_up`, `aeris_model_ready`; all 1 | Any 0 for 15 seconds; no request minimum | Project operator: check process/UI health and bundle hashes; restore verified selection. Never treat exporter UP alone as ready |
| Fixed-reference source shift (R4/R6) | `aeris_source_shift_fraction`; fraction of temperature entries beyond 3 training-source standard deviations, averaged over last 100 eligible requests; expected near source replay baseline | >0.20 for 30 seconds and `aeris_shift_support >= 20`; provisional, not an accuracy guarantee | Model reviewer: investigate sensor units/context and retained raw-window audit; keep unsupported-setting review; retest any mitigation |
| Review workload (R1/R5/R6) | `increase(aeris_review_total[5m]) / increase(aeris_requests_total[5m])`; expected depends on setting mix | >0.80 for one minute, at least 20 requests in five minutes; provisional | Model reviewer: inspect setting coverage, disagreement and shift. Do not weaken gates to make the dashboard green |
| Feedback coverage/quality support (R7) | Feedback/eligible predictions, agree/(agree+disagree), overrides/reviewed predictions, satisfaction sum/responses, independently inspected labels/eligible predictions | No hard quality alarm without support. Unsure is separate; zero denominators return no data. Investigate sustained low coverage after real adoption | Project operator: follow up on inspection workflow. Model reviewer: verify provenance and independent case counts before training |

All fractions are 0–1; latency is seconds and throughput is requests/second.
Histogram p50/p95 use rate of the persistent inference duration buckets. These
measure server inference handling and ledger request scope, not browser paint.
The direct 100-request timing is end-to-end HTTP, also excluding browser paint.
The original 95-of-100 rendered UI results under five seconds target therefore
remains unverified, even when the API passes it.

The `test` and `genuine` cohorts are explicit labels and separate queries.
Setting, architecture and class labels use fixed bounded vocabularies.
Prediction IDs, raw sensor values, inspection identities and free text remain
in the ledger and never appear in Prometheus labels. Counter totals derive
from unique durable prediction rows, so restart or UI rerun does not inflate
them. Feedback edits replace the current per-prediction response and retain a
separate history; feedback counts are gauges, not event counters.

Group monitoring exposes request/review counts and verified class support by
setting. Outcome recall is not displayed as validated performance when
independent-case support is absent. Current genuine verified-label support is
zero. Retrospective dataset labels and simulated reviews are never substituted
for live outcomes. Compare raw numerators and denominators before interpreting
any apparent subgroup gap.

`failure_demo.json` records an actual missing-bundle alert, recovery, and a
persistent-feedback restart check. `monitoring_queries.json` and
`grafana_live_query.json` contain actual backend query results. Dashboard
provisioning is verified via Grafana's API; browser visual verification remains
separate and currently incomplete because the computer-use safeguard stopped it.
