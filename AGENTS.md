# AERIS project requirements

Read this file before changing training, adaptation, evaluation, or UI code.

## Confirmed user requirements (2026-10-22)

- AERIS is one integrated AHU diagnosis project: hybrid modeling, V0/V1/V2,
  cross-setting adaptation, confidence handling, technician-verified feedback,
  comparisons, and the operator interface belong to the same project.
- Every version combines classical ML (currently Random Forest) and a temporal
  neural network into the final predictor.
- V0 is the baseline. V1 and V2 evolve its architectural design and each requires
  its own trained hybrid model. Architecture evolution does not mean automatically
  inheriting V0's trained weights. Initial V1/V2 training starts fresh.
- Do not equate V0/V1/V2 with source-only / pseudo-label / feedback stages.
  Cross-setting adaptation remains part of the integrated workflow.
- The requested temporal training budget is 50 epochs. Explicitly document any
  early-stopping policy and distinguish maximum epochs from completed epochs.
  Random Forest fitting uses trees, not neural-network epochs.
- Preserve the common data split and held-out evaluation protocol for comparisons.
  Introducing data again for fresh training does not authorize fitting on test data.
- Keep requirements.txt current whenever dependencies change.
- Historical Windows interpreter (the user now requests autonomous execution rather than PowerShell instructions):
  C:\Users\mhars\miniconda3\envs\aeris-data\python.exe
- First deliver a working milestone demonstration with the UI and cross-adaptation;
  subsequent submissions improve the versions and repeat comparisons.

## Verified implementation state

- V0 uses a Random Forest and a causal TCN, with a validation-selected fixed
  probability mixture. V0 has completed training on the user's machine.
- scripts/05_adapt_v1.py is a legacy implementation based on the wrong version
  mapping. Its previous run skipped training due to insufficient candidates.
  Neither that skipped run nor a copied V0 bundle counts as architecture V1.
- Preserve legacy results for traceability; do not overwrite or delete them
  when installing the corrected architecture implementation.
- The user selected the proposed V1 design on 2026-10-22: freshly trained Random
  Forest plus a freshly initialized causal TCN with learned attention pooling
  across the 15-minute window, combined into the hybrid prediction. Retain V0's
  RF settings and encoder size to isolate the pooling change. Run 50 epochs by
  default, with early stopping disabled, and retain the best validation checkpoint.
- V1 is saved in v1_attention; the legacy v1 folder is historical evidence only.
  Cross-adaptation retains the architecture version and writes a suffixed bundle.
- Verified 2026-10-26: V1 completed all 50 epochs and selected checkpoint 18.
  V0 completed 37 epochs and selected 29. The uploaded measured comparison uses
  the same 6,930 source-validation windows: macro F1 V0 0.6886649651, V1
  0.6884134584; accuracy V0 0.7610389610, V1 0.7466089466. Attention improved
  normal recall but did not demonstrate an overall diagnostic improvement.
- Today's First Model Exploration evidence is in experiments/model_exploration/.
  Read REPORT.md for actual data separation, errors, timing scope and limitations.
  The full UI target and successful real-target adaptation remain unverified.
- On 2026-11-02 the user confirmed their professor requires the complete pipeline
  and a working UI BEFORE filling the Milestone 4 documentation. This is the
  implementation target even where the generic template has later dates.
- On 2026-11-02 the user selected V2: freshly trained RF + a fresh TCN combining
  final-step and attention-pooled representations. Keep encoder inputs/settings
  fixed; the combined temporal head is the architectural change.
- M4 comparisons use separate m4_v0/m4_v1/m4_v2 bundles, fresh training for 50
  epochs, patience 0 and fixed 0.50 RF fusion. Keep all earlier bundles intact.
- Professor's M3 feedback must affect implementation and documentation. Retain
  original m3-v1 and its five-independent-case reporting floor. Additional windows
  or severity variants are not additional independent conditions. Current source
  experiments are descriptive, within-recording development studies of the three
  shared-weather groups; independent generalization claims remain blocked.
- M3 records that target labels were exposed earlier. Existing target files are
  exploratory, not a newly sealed holdout. Channel units/roles/modes and command
  semantics require explicit checks; high model confidence cannot override scope.
- The bounded clean-data check passed on Windows on 2026-11-02: exact target ZIP
  checksum/CRC, 91 parsing failures reproduced, m3-v1 unchanged. It does not verify
  a fresh full CUDA environment or a fresh large-source download.

These requirements supersede the conflicting version mapping in older project
documentation. Source documents describe earlier scope; the user's latest
explicit requirements govern implementation.

- Autonomous execution now uses a separate Linux Python 3.11.17 CPU environment,
  with exact raw manifest and split records matched to the historical project.
  Current requirements.txt pins portable CPU training + Streamlit dependencies;
  requirements-cuda.txt retains the Windows CUDA alternative.
- Do not declare target transfer successful when gates skip an update or a source
  retention guard rejects it. Published-reference feedback is explicitly simulated
  technician feedback, not independent human inspection.
- The user requested completion and thorough verification of the pipeline and UI
  before beginning the M4 Word document. Do not fill that document prematurely.
