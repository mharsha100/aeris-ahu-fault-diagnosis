# V1 architecture training and cross-setting adaptation

## Confirmed design

On 2026-10-22 the user selected a fresh Random Forest plus a fresh causal TCN
with learned attention pooling over the 15-minute input window. This is V1.
V0/V1/V2 are architecture versions within one AERIS project. Cross-setting
adaptation, verified feedback, comparison and the operator UI remain in scope.
The exact V2 architecture is still to be specified.

V0 reads the final TCN time-step representation. V1 scores each of the 15 encoded
time steps, normalizes those scores with softmax, and classifies their weighted
sum. The scores are learned during neural training. They are not verified causal
explanations of a fault. Every input time step precedes or equals the prediction
endpoint; the encoder uses causal convolutions.

RF settings (200 trees, depth 18, minimum leaf 3, balanced subsample weights),
TCN channels (32), dilation blocks (1/2/4), dropout (0.15), features and prepared
splits stay consistent with V0. This makes attention pooling the main structural
change. As in V0, a source-validation-selected probability mixture combines both
branches, followed by source-only calibration. The RF is fitted separately from
the neural optimizer; this is a hybrid predictor, not backpropagation through trees.
A fresh RF can produce identical predictions to V0 when settings, seed and data
are identical. That does not mean its saved weights were reused.

V1 initialization never loads V0's forest or neural weights. V0's metadata verifies
that the input packs, scaler and protocol match. Training uses the existing
source_train pack; there is no new random split or use of held-out test labels.

## Training policy

The default is 50 complete neural epochs, early stopping disabled, retaining the
best source-validation checkpoint. RF fitting is measured in trees, not epochs.
Pass a positive --patience to enable early stopping deliberately. The actual V0
run stopped at 37 epochs and selected epoch 29; disclose that budget difference
in comparisons. A later controlled comparison can use matched training budgets.
More complexity does not guarantee improved performance.

## PowerShell commands

Run from the existing aeris-data-warmup_updated project directory:

```powershell
$AerisPython = "C:\Users\mhars\miniconda3\envs\aeris-data\python.exe"
& $AerisPython -m pip install -r .\requirements.txt
& $AerisPython -m pip check
& $AerisPython .\scripts\05_train_v1.py --device cuda --check-only
& $AerisPython .\scripts\05_train_v1.py --device cuda --epochs 50 --patience 0
```

The check-only command checks the runtime and attention-network forward pass.
Full data/provenance checks happen at the start of training. Stop at any error.

Output: data/aeris_models/v1_attention/. The folder contains the trained forest,
neural checkpoint, configuration, calibration, training history and source
validation reports. v1_review.zip contains reports only and can be shared for
review. An existing output is protected against accidental replacement.
data/aeris_models/v1/ is retained as evidence of the old skipped attempt; it is
not the new architecture. The legacy 05_adapt_v1.py command is disabled.

## Cross-adaptation in the same AERIS workflow

After checking the initial training reports, the implemented next command is:

```powershell
& $AerisPython .\scripts\06_cross_adapt.py --model v1_attention --device cuda --epochs 5
```

This additional five-epoch adaptation budget is separate from the completed
50-epoch initial fit. It audits target_adapt using the trained V1 model, fits RF
with source replay and gated pseudo-labels, and fine-tunes that same attention
architecture. Source-retention checks determine whether to retain the update.
The model version remains V1. Results go to v1_attention_cross_adapted/.
The cross_adaptation_review.zip report explicitly records whether training was
attempted, how many epochs ran, and whether an update was retained. Insufficient
candidates can still result in a skipped adaptation; that does not invalidate
the separately completed V1 architecture training or demonstrate successful transfer.

The same command supports --model v0 for the baseline, writing v0_cross_adapted/.
Both use the same fixed gate policy and target_adapt pool. Target reference labels,
feedback data and final test packs are not used by these commands. Operator notes
are not verified labels. The technician-feedback update and V2 implementation
remain subsequent project work, alongside final comparisons and the UI.

Compare saved source-validation results after initial V1 training, and again after
adaptation, with:

```powershell
& $AerisPython .\scripts\07_compare_development.py
```

This writes data/aeris_comparison/development_comparison.zip with class-level
metrics, initial/adaptation epoch counts and adaptation status. It includes any
completed cross-adaptation bundles and does not open final test inputs.

## Interpretation and limitations

Source-validation scores are development results, not final target-domain scores.
Source retention is not proof of target improvement. The current split holds out
weather days within the same simulated scenarios, rather than unseen buildings
or independent fault injections. Final cross-setting comparisons must use the
frozen held-out evaluation protocol and report that limitation.

requirements.txt retains the versions verified on the Windows/GPU V0 run. V1 and
cross-adaptation need no additional package. The CUDA build is specific to that
environment; local developer checks use the corresponding CPU PyTorch build.
