# V1 update verification

Code verification uses synthetic source fixtures and PyTorch 2.9.1+cpu with
NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3 and scikit-learn 1.7.2. It does not
establish project accuracy or verify the laptop's CUDA execution.

Verified behaviors:

- Original V0 bundles without the new architecture metadata remain loadable.
- Fresh V1 training succeeds with V0's weight files deliberately unreadable;
  V0's provenance files alone are consulted to check the comparison protocol.
- A 12-epoch test run with early stopping disabled completes all 12 epochs.
- The learned attention receives gradients and its time weights sum to one.
- The temporal encoder preserves causal prefixes.
- Saved/reloaded V1 predictions match in-memory predictions.
- Existing initial model output is protected, and changed source packs are refused.
- Corrupt test/reference/feedback inputs do not affect training or adaptation,
  demonstrating those inputs are not opened along the exercised paths.
- Target adaptation covers insufficient-candidate, accepted-update and rejected-
  update paths. Accepted/rejected fixtures use synthetic target-like inputs.
- A rejected update restores the original predictor. Parent artifacts stay intact.
- Cross-adaptation retains V0 or V1 identity and correctly reloads the attention model.
- Development comparisons check data provenance and report actual training epochs.

The Windows run must still check the attention forward pass on CUDA and train on
the user's actual prepared source packs. The review ZIP from that run provides
the project evidence. The user's existing V0 trained 37 epochs with an epoch-29
checkpoint; V1's default is 50 epochs. These budgets must be disclosed when comparing.
