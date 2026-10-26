# V0 review: first Windows/GPU training run

Run completed 2026-10-22 using Python 3.11.9, PyTorch 2.9.1+cu128 and the
NVIDIA GeForce RTX 4070 Laptop GPU. Source fitting used 27,720 windows;
validation and calibration each used 6,930 windows. Training stopped at epoch 37
and restored temporal epoch 29. The selected fusion was 75% Random Forest and
25% temporal network. Model saving/reloading reproduced probabilities within
2.3e-16 absolute difference.

| Component | Source validation accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Random Forest | 0.7625 | 0.6884 | 0.4947 |
| Temporal network | 0.6843 | 0.6674 | 0.4933 |
| Hybrid V0 | 0.7610 | 0.6887 | 0.4731 |

The hybrid's classification gain over the forest is negligible. Its probability
metrics improved on these source validation days. This does not establish a
general advantage for hybrid models or successful transfer to real AHUs.

The hybrid's per-class F1 scores were normal 0.3295, damper 0.8031, cooling valve
0.8021 and heating valve 0.8199. Normal recall was 214/462 = 0.4632.
Every normal validation window was flagged for review. The provisional confidence
and branch-agreement rule accepted 3,610/6,930 windows (52.09%); those accepted
windows had no observed validation classification errors, but included no true
normal examples. This is limited development evidence, not a target error bound.

Normal versus fault separation needs investigation. These metrics alone do not
identify whether observability, class imbalance, feature coverage or model
capacity is responsible. V0 remains fixed while the initial adaptation and
feedback versions are built.

Validation consists of three held-out weather-day groups within existing
simulated scenario runs. Windows within a day overlap and are dependent.
Model/fusion choices used validation; these are development results. No
source final test, target adaptation, feedback or target final test was used
by this V0 fitting run.

Source: the user's v0_review.zip and matching terminal output.
