# Text-box submission

Commit identifier and commit-pinned links will be supplied after publication.

**AERIS — First Model Exploration**

The AI component identifies normal operation, damper fault, cooling-valve fault or heating-valve fault from 15-minute occupied/fan-on AHU sensor windows. It produces class probabilities and a likely condition; deterministic confidence/disagreement rules route uncertain cases to human review. This exploration covers supervised source-domain diagnosis first, which establishes a measurable diagnostic component before cross-setting adaptation and the UI are integrated.

Candidates considered were Random Forest, a causal temporal convolutional network (TCN), and an RF+TCN hybrid. The hybrid was selected to combine window statistics with temporal patterns on the available RTX 4070 laptop. Unit conversion, schema checks, filtering, time grouping, feature calculations and review rules are non-AI stages. The temporal encoder and RF are trained separately and combined into one predictor.

Inputs are 45 documented G36-1wk simulated runs: ten core channels, 15 one-minute samples per window, stride five minutes, and 66 RF summary features. The source split contains 27,720 training, 6,930 validation and 6,930 calibration windows. Validation class counts are normal 462, damper 3,234, cooling 2,156 and heating 1,078. Missing/invalid values are flagged rather than imputed; invalid or out-of-scope rows do not enter windows. Units are aligned to Celsius/fractions. Standardization uses source training only; probability calibration uses the separate calibration partition. Final source and real-target tests remain unused. The implemented split holds out weather days within the same scenarios, rather than entire independent scenarios/buildings; overlapping windows are dependent. Data contracts, manifests, quality audit and file hashes are included in the report.

Both runs use seed 42, 200 RF trees (depth 18, minimum leaf 3), a 32-channel causal TCN with dilations 1/2/4, AdamW learning rate 0.001, batch size 256 and class-weighted loss. Macro F1 is primary because the classes are imbalanced; per-class recall/F1 and confusion matrices are also reported.

| Run | Configuration / change | Epochs completed; best | Validation N | Macro F1 | Accuracy | CPU inference p95 |
|---|---|---|---:|---:|---:|---:|
| V0 | RF + causal TCN, final-step pooling; RF fusion 0.75 | 37; 29 | 6,930 | 0.688665 | 76.10% | 22.28 ms |
| V1 | Fresh RF + TCN with attention pooling; RF fusion 0.50 | 50; 18 | 6,930 | 0.688413 | 74.66% | 16.63 ms |

The targeted change was replacing V0's final-step temporal pooling with learned attention across the window, with the hypothesis that a fuller temporal summary could help weak normal/fault discrimination. Both V1 branches were trained fresh on the same data. The starting optimizer, RF/encoder settings, seed and scoring procedure were held fixed. V0 stopped at 37 epochs (best 29); V1 completed 50 (best 18). The same fusion-selection rule selected RF weights 0.75 and 0.50. These budget/fusion differences are disclosed, so the result is not an isolated causal test of attention.

Twenty errors were inspected, ten per model, with full window IDs and conditions in the report. Examples include V0 shoulder-season cooling fault `0cf0800a4efb...` predicted as damper fault at 0.8068 confidence; normal window `b6d33352232c...` predicted as damper fault at 0.7442; and heating fault `4b4047f4ef8f...` predicted as damper fault at 0.6906. The same cooling window `dfa4aad727bf...` is wrong in both versions, while confidence rises from 0.7764 to 0.8405. V1 cooling window `039aa7408963...` is predicted as normal at 0.6284. Branch agreement therefore does not establish correctness. All inspected errors route to review, and no incorrect validation prediction meets the 0.90 confidence threshold; this is not a guarantee on real target data.

V1 improves normal recall from 46.32% to 59.31%, but damper recall falls from 82.50% to 76.47%. Fault-to-normal predictions increase from 623 to 916. Macro F1 is nearly unchanged and accuracy decreases by 1.44 percentage points. The change does not demonstrate an overall diagnostic improvement. No statistical significance or real-target benefit is claimed.

Timing used the same 100 stratified validation windows, three warmups, CPU inference, four torch threads and one RF job. All 100 requests per model were below five seconds. This measures input checks, inference, calibration and review decisions; it excludes loading, raw preprocessing and UI rendering. The Section 2.1 goal of at least 95/100 full UI requests within five seconds remains unverified. Summed neural epoch training/validation time was 127.91 s for V0 and 197.29 s for V1, excluding setup, final calibration and saving.

Next steps are a matched-budget/fusion comparison, raw-trace inspection of the false-normal and cross-fault confusions, and continued V0/V1/V2 development, cross-setting adaptation, verified feedback and the UI within the same AERIS project. Organized scripts and saved outputs are provided instead of a new training notebook. The complete report records candidates, data provenance, configurations, per-class metrics, all reviewed errors, timing conditions and limitations. Dataset releases are attributed; raw data, credentials and weights are excluded from GitHub.
