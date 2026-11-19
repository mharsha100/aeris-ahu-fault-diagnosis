# AERIS corrected learning workflow and demonstration guide

This guide supersedes the V0/V1/V2 explanation in the earlier project guide. It explains the corrected implementation in plain language. Historical architecture experiments remain evidence of earlier work; they are not the live learning stages.

## What the three versions now mean

Think of the model as a trainee learning to read an air-handling unit's sensor history.

**V0 learns from the original examples.** It reads a 15-minute window and estimates one of four conditions: normal operation, damper fault, cooling-valve fault or heating-valve fault.

**V1 learns from confident guesses.** When an eligible new input passes the confidence, component-agreement and sensor-range checks, the model's predicted class becomes a temporary training answer. This is called a pseudo-label. It is a guess, not a technician-confirmed truth. We require at least 95% confidence for this queue, in addition to the existing model review checks.

**V2 learns from reviewed answers.** An uncertain input asks for a technician. A confirmed class, reviewer identity and evidence connect the answer to that exact input. The background trainer uses the reviewed answer together with the input. The original source data are replayed during training so the model does not learn only from a handful of new examples.

You submit each input once. You do not choose three different architectures. The application automatically uses the latest approved learning stage for the selected cohort. Predictions retain their original model and audit ID even after a later stage is activated.

Training happens in batches of eight new or corrected windows. A guess does not instantly turn into a new trained model. While training runs, the previous approved model continues answering requests. V2 may be trained directly from V0 when no V1 candidate is acceptable; the saved parent history makes that explicit.

The source model combines Random Forest, a temporal neural network, and CatBoost. These are model components. They are separate from the names V0, V1 and V2. The old artifact name boosted_v2_s42 describes its historical architecture; in the corrected lifecycle it supplies source-only V0.

## What actually changes during learning

The Random Forest is fitted again with source examples plus the accumulated adaptation examples. The temporal network is fine-tuned for up to 50 epochs, with a documented early-stop rule that restores retained weights if source performance falls. CatBoost receives 120 additional boosting iterations and preserves the original class-balancing policy. Probabilities are recalibrated on the source calibration split.

A candidate must retain source performance relative to both its parent and the original V0: macro F1 may fall by at most 2 percentage points, and no class recall may fall by more than 5 points. These are provisional research guardrails. Passing them does not prove that the candidate is better at a new building. A rejected candidate remains in history; it does not replace the active model.

The first V1 attempt exposed a continuation bug: it did not preserve CatBoost's SqrtBalanced class weighting. Overall F1 rose slightly but normal-class recall fell by 14.29 points, so promotion was rejected. The corrected retry preserved class weighting and reused the identical completed Random Forest and 50-epoch temporal fit. The report distinguishes reused training from newly executed training. The interrupted V2 attempt remains visible; the subsequent V2 run uses the approved parent.

## What the measured scores mean

Accuracy is the fraction of all predictions that are right. Precision asks: when we name a particular fault, how often is that answer right? Recall asks: among examples that truly have that fault, how many did we find? F1 balances precision and recall. Macro F1 gives each of the four classes equal weight, so a common fault cannot hide a neglected class.

Confidence is the model's own estimated probability for this input. It is not an accuracy measurement. A model can be very confident and wrong. Calibration checks whether confidence roughly matches observed correctness on a specified dataset; lower calibration error is better.

Before this lifecycle correction, the selected source model measured 78.07% accuracy and 70.67% macro F1 on source validation, and 21.31% macro F1 on the retrospective target test. The target result is weak. High-quality tabular files do not remove differences between buildings, operating conditions, sensors and fault behavior.

The executed attempts below are source validation results plus retrospective target reporting. Target test labels did not enter fitting, calibration, early stopping or the promotion decision. These target records were already exposed in earlier project work, so this is not a new blind field trial.

| Attempt | Outcome | Source accuracy | Source macro F1 | Target macro F1 |
|---|---|---:|---:|---:|
| V1 52ef6dd8 | rejected | 79.45% | 70.71% | 21.66% |
| V1 41511bc5 | promoted | 78.14% | 70.78% | 21.70% |
| V2 87a6cefd | promoted | 78.04% | 70.75% | 25.73% |
The approved V2 source-validation precision is 70.19%; recall is 72.37%; calibration error is 2.29%. These precision/recall values average the four classes equally. Target accuracy is 37.24%, so target performance is still insufficient for a production-accuracy claim. V2 source F1 is slightly below V1, while target F1 improved. The versions do not improve every metric monotonically.

Final V2 library evaluation on the same 40 balanced source windows:

| Temperature budget | Command budget | Clean accuracy | Attacked accuracy | Attack success among initially correct | Accepted wrong |
|---|---|---:|---:|---:|---:|
| +/-0.2 C | +/-0.01 | 80.0% | 65.0% | 18.8% | 0 |
| +/-0.5 C | +/-0.02 | 80.0% | 62.5% | 21.9% | 0 |
| +/-1.0 C | +/-0.05 | 80.0% | 60.0% | 25.0% | 0 |

V2 SHAP checks: 20 cases; repeatability 100%; strongest-group stability 95%; maximum additivity error 7.67e-10. These are model-explanation checks, not independent technician ratings.


## Which attack library we use

We now use **Adversarial Robustness Toolbox 1.20.1**, commonly called ART. The implemented attack is **HopSkipJump**, through ART's BlackBoxClassifier. It queries the complete predictor, including Random Forest, the temporal network and CatBoost. We do not pretend that attacking only the neural branch measures the whole system.

The attacker makes small bounded changes to temperatures and control commands. Fan state, occupancy and timestamps stay fixed. The test uses 40 source windows, 10 from each class. This is a finite-budget experiment; an attack that did not find a failure has not proved no failure exists.

The original source model got 30 of these 40 cases right before attack. ART changed 6 of those 30 at the smaller two budgets, and 7 of 30 at the largest budget. Thus attack success was 20%, 20% and 23.33%. Raw attacked accuracy was 60%, 60% and 57.5%. The review policy allowed **zero wrong predictions to pass without review in this test**. That is evidence about this test, not a guarantee against every attack.

ART here tests model evasion through sensor inputs. It does not scan Docker, the network or the entire application for every security vulnerability. API validation, artifact checks, durable audit records and recovery tests cover separate system controls.

Earlier experiments used custom NumPy perturbations. Those results remain historical; they are not relabeled as ART results.

## Which explanation library we use

We now use **SHAP 0.51.0**, specifically **ExactExplainer**. SHAP is suited to this tabular and time-window predictor. Grad-CAM is designed around neural activation maps, commonly for images; it would not explain this entire RF plus temporal plus CatBoost system.

The explanation asks how four sensor groups move the predicted class probability: temperatures, ventilation controls, cooling command and heating command. It evaluates all 16 combinations of those four groups. A group that is removed is replaced with source-training averages. Fan and occupancy stay fixed.

Example: a cooling-command contribution of +38 percentage points means that group helped move the model toward the cooling-fault prediction relative to the reference. It does not prove the valve physically caused the fault. A negative contribution moves the model away from that prediction.

On 20 source-only V0 cases, repeated SHAP explanations matched in 100% of checks. The strongest group stayed the same in 90% of checks after a +0.05°C temperature change. The maximum additivity error was about 5.26e-10 probability units, essentially numerical rounding. Additivity means the explanation's contributions add back to the model's output. It is not a 100% score for human usefulness or physical truth.

Earlier explanations used a custom grouped Shapley implementation. The live endpoint now calls the SHAP library.

## What the demo data mean

The V1 rehearsal used the eight eligible confident windows found among 1,420 target adaptation windows. They are not eight independent buildings. Most target windows failed the strict confidence/review gate. That means limited self-training support, not that the files are necessarily poor quality.

The V2 rehearsal selected 20 non-overlapping target-feedback windows using model uncertainty before revealing their published reference labels. Those labels simulate a technician for demonstration. They are explicitly marked test data; no real human inspection is claimed. Genuine operator learning has a separate model registry and is not altered by the demonstration.

Source validation, source calibration, source test and target test are protected evaluation roles. They are never queued by the recorded-input API. Known exported evaluation windows are also excluded when uploaded again. Repeated requests and repeated sensor windows do not duplicate training examples. A technician correction replaces a pseudo-label for the same window.

## How to show the professor

1. Open http://127.0.0.1:8510/. Leave the cohort on Test / demonstration. Explain the V0 → V1 → V2 learning card. It shows the active stage and the latest training outcome.
2. Choose Adaptation inputs and window 27, then click Analyze window once. The saved result shows the stage that actually answered, probabilities, review status and an audit ID. An already-used window is deduplicated rather than trained again.
3. Click Explain full-hybrid prediction. Read the SHAP bars as factors increasing or decreasing the model's probability. Do not call them proof of the physical cause.
4. Choose Technician learning inputs and a window. An uncertain case asks for review. For a demonstration, use the documented reference answer and clearly identify it as simulated. Enter the technician's class, reviewer name and evidence; Save review. Ratings and agreement alone do not train.
5. Explain that eight new or changed eligible windows start a background batch. The existing model stays available. The learning card reports training, promotion or rejection; a finished run does not automatically mean improvement.
6. Open http://127.0.0.1:9091/targets. Expand the aeris pool if needed. The endpoint is http://app:8001/metrics and its state should be UP. This means Prometheus can collect measurements, not that model accuracy is high.
7. Open http://127.0.0.1:3001/d/aeris-operations for request activity, latency and review workload. Open http://127.0.0.1:3001/d/aeris-learning for stage history, trainer health, learning samples and candidate outcomes. Keep the cohort on test. Prometheus scrapes every 5 seconds; Grafana refreshes every 10 seconds. Allow a scrape and refresh after a request.
8. Open Model evidence and Risk & assurance in AERIS. Show the actual model scores, failed candidates, ART attack results and SHAP checks. Explain that the historical architecture comparisons are separate from the corrected learning stages.

The setting-distribution chart counts submitted requests. If many tests used CoolingSeason, its bar is large. It does not describe the class balance or quality of the training dataset. Missing live accuracy means there are no independently verified field outcomes; it must not be replaced with an invented number.

## What we can responsibly claim

We have an integrated supervised research deployment with real training, conditional model promotion, complete-model SHAP explanations, ART attack measurements, persistent reviews, Docker services and live monitoring. Test and genuine learning are isolated. The deployment does not control equipment.

We cannot claim universal attack resistance, guaranteed improvement with every update, independently established target-building accuracy, or complete production readiness. Wider deployment still needs independent site evaluation, authenticated access, transport security, an operational owner and more representative reviewed cases. A successful local demonstration is useful evidence of engineering progress; it is not those missing field studies.

Library references: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html and https://shap.readthedocs.io/en/stable/generated/shap.ExactExplainer.html.
