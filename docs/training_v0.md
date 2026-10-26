# AERIS first hybrid implementation

V0 combines a Random Forest trained on 66 window summaries with a causal temporal
convolutional network trained on the corresponding 15-by-10 sensor sequences.
Both components contribute to one final class-probability vector. The initial
operating scope is occupied periods with the AHU supply fan running.

## Data use

- `source_train`: fits the forest and temporal network.
- `source_validation`: selects the temporal checkpoint and fusion weight.
- `source_calibration`: fits scalar probability-temperature parameters.
- Source and real target final test packs remain unopened during V0 training.
- Real adaptation and feedback packs remain unopened during V0 training.

The source split holds out weather days across all simulated scenarios and their
baseline. It evaluates new days within existing fault scenarios, rather than
unseen buildings or independent fault injections. The common real test set is
reserved for comparing frozen V0, V1 and V2 models.

## Initial configuration

The Random Forest uses 200 trees, maximum depth 18, a minimum of three samples per
leaf and balanced bootstrap class weights. The temporal network uses 32 channels,
three residual blocks with dilations 1/2/4, kernel size three, per-time-step layer
normalization and dropout 0.15. Its final time-step representation predicts the
four classes. AdamW uses learning rate 0.001, weight decay 0.0001 and class-weighted
cross-entropy. Maximum training is 50 epochs with source-validation early stopping.

The fusion candidates give the forest weights 0.25, 0.50 or 0.75; the temporal
weight is the complement. Source-validation macro F1 selects the weight, with
log loss and proximity to equal weights breaking ties. Calibration is fitted
using source calibration days; source calibration does not establish reliable
confidence on the real domain. The initial review rule uses confidence below
0.90 or branch disagreement and must be evaluated before operational use.

## Environment and execution

Python 3.11.9 and the existing pandas 2.2.3 environment are retained.
The successful Windows/GPU run used NumPy 2.4.6, scikit-learn 1.7.2,
SciPy 1.16.3, joblib 1.5.2 and PyTorch 2.9.1+cu128. These versions are now
pinned together in requirements.txt, including the official CUDA 12.8 index.
requirements-training.txt forwards to that file for compatibility:

```powershell
& "C:\Users\mhars\miniconda3\envs\aeris-data\python.exe" -m pip install -r requirements.txt
& "C:\Users\mhars\miniconda3\envs\aeris-data\python.exe" scripts/04_train_v0.py --device cuda
```

Official installation reference: https://pytorch.org/get-started/previous-versions/

`--check-only` runs an actual temporal forward pass on the requested device.
`--device cpu` supports CPU execution. `--overwrite` intentionally replaces an
existing V0 bundle. V1/V2 identify architecture versions; cross-adaptation keeps
the selected architecture identity and writes a suffixed bundle. See training_v1.md.
The pinned package list captures the successful GPU environment, including its
CUDA build; it is not a complete lock of every transitive notebook dependency.

## Outputs

`data/aeris_models/v0/` contains both trained components, their shared feature
contract and scaler, fusion/calibration configuration, training history and source
validation reports. Checkpoint reloading is checked against in-memory predictions
before the final model bundle is published. `v0_review.zip` contains small reports
for inspection; it excludes model weights and training arrays.

`src/aeris/modeling.py` provides `load_hybrid` and the shared inference interface.
`src/aeris/features.py` reproduces the frozen tabular and sequence transformations
for later inference. These components are intended to be reused for V1, V2 and
the user interface. No test or real adaptation score is claimed by this step.
