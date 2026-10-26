# Historical implementation: superseded version mapping

**Do not follow the commands below.** This document records the legacy attempt
that incorrectly used V1 to mean adaptation of V0. The actual user run skipped
training. V1 now means the independently trained attention-pooling hybrid; see
`training_v1.md`. Use `06_cross_adapt.py` for adaptation of a selected trained
architecture within the same AERIS project. The old command is disabled.

## Original implementation record

V1 keeps the same Random Forest + temporal neural architecture, the source
scaler and the V0 fusion weights. It audits only target_adapt inputs, then
attempts one adaptation round. It never opens the feedback pool, either final
test pack, or target reference labels.

## Fixed selection rules

A candidate needs hybrid confidence at least 0.95, each branch's probability
for the same class at least 0.90, branch agreement, and all standardized
features within six source standard deviations. This last rule is a broad
plausibility filter, not proof of domain overlap.

The gates and predicted class must hold for three consecutive five-minute-stride
windows within the same continuous run/day. Selected training windows do not
share minutes with one another. Selection is deterministic and capped at eight
windows per day, 24 per predicted class and 16 per mode/class. At least 12 selected
windows, two days and two predicted classes are required to attempt training.
All these labels are unverified model guesses.

## Training and retention

The forest is refitted using every original source-training window and the
selected pseudo-label windows. Source rows have sample weight one; target rows
have weight five times the minimum of their hybrid and branch confidences.
Forest hyperparameters and random seed are retained from V0.

The temporal branch starts from V0 and uses five fine-tuning epochs at learning
rate 0.0001. Every epoch replays the full source training set. The loss is
90% class-weighted supervised source loss plus 10% confidence-weighted target
pseudo-label loss. Target mini-batches are balanced across the selected predicted
classes. If temporal source-validation macro F1 drops more than 0.02 from V0,
training stops and restores the last retained temporal state.

Branch and hybrid probability temperatures are refitted using source calibration
only. A candidate hybrid must keep source-validation macro F1 within 0.02 of V0
and every class's source recall within 0.05. These thresholds are fixed prototype
source-retention limits, not evidence that target predictions improved.

## Status meanings

- adapted: the trained candidate passed the source-retention limits and was saved.
- skipped_insufficient_candidates: there was insufficient candidate coverage; the
  V1 bundle preserves V0 predictions.
- rejected_source_validation: the candidate failed the source-retention limits;
  the V1 bundle preserves V0 predictions.

The retained temporal adaptation epoch is reported; zero means the temporal
weights reverted to their V0 state. A forest update can still be retained if
the combined candidate passes the final source check.

V0 model files are never replaced by this step. V1 is saved to data/aeris_models/v1
and reload-checked using the same shared inference interface as V0.

## Run

The V0 dependencies are sufficient; there are no new V1 packages. The main
requirements.txt is the single version list and includes the verified CUDA
PyTorch build. Install/update with:

    & "C:\Users\mhars\miniconda3\envs\aeris-data\python.exe" -m pip install -r requirements.txt

From the project root, run:

    & "C:\Users\mhars\miniconda3\envs\aeris-data\python.exe" scripts/05_adapt_v1.py --device cuda

The small data/aeris_models/v1/v1_review.zip contains the gate audit, source
retention results, training history and model configuration. It excludes model
weights and target ground truth. The frozen V0 and all reviewed window files
must still be present.

Future comparison uses the common held-out target test after V0/V1/V2 are
frozen. V1 being saved, or source metrics staying stable, does not establish
successful cross-domain adaptation.
