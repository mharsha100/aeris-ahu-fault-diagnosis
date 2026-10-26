# First Model Exploration — measured V0/V1 results

Read **[REPORT.md](REPORT.md)** for the complete assignment report, configurations,
specific errors, timing conditions, comparison limits, reproduction instructions
and requirement-to-evidence mapping.

- [Experiment table](experiment_results.csv)
- [V0 measured evidence](v0/)
- [V1 attention measured evidence](v1_attention/)
- [Data preparation and split audit](data_audit/)
- [Experiment scripts](../../scripts/)

Macro F1: V0 **0.688665**, V1 **0.688413**, on the same **6,930** source-validation
windows. CPU inference p95: **22.28 ms** and **16.63 ms**, respectively, on the same
100 development windows. Attention improves normal recall but does not improve
overall macro F1/accuracy in this initial comparison. These are development scores,
not final target-domain results. UI latency and target transfer remain unverified.
