# AERIS implementation checkpoint

This folder versions the M4 implementation and evidence. The completed, reviewed working document is preserved in `docs/AERIS_Milestone4_AI_Component_and_Experimentation.docx`. Its access-status notes describe the state before this Git Bash publication; publication must be verified before those notes and repository links are revised.

The current achievable question is a descriptive comparison of diagnostic errors, class recall and review workload within the selected recordings. Forty-five source runs share only three seasonal weather conditions. Extra fault severities and overlapping windows do not create independent cases. The original five-independent-case rule from m3-v1 remains unchanged, and broader class-by-season generalization claims remain blocked. The two published target files were previously explored; their scores are retrospective and exploratory.

## Architecture and stage

Each initial architecture independently fits a Random Forest and initializes and trains its own temporal network. All receive the same frozen input packs, RF200, seed42, batch256, 50 complete epochs, no early stopping, fixed .50/.50 fusion and source-only calibration. V0 uses the last encoded state; V1 learns attention across the observed window; V2 combines last-state and attention summaries with a learned head. Selected checkpoint epochs and parameter counts are in the actual training records. One seed is a development comparison, not evidence of robustness across seeds.

Architecture version remains V0/V1/V2 through separate source, cross-setting adaptation and verified-feedback stages. Cross-setting gates never use target labels. Insufficient pseudo-label coverage skips fitting and retains the parent. Feedback IDs are frozen before their selected public reference labels are revealed, and source replay/retention guards decide whether to retain the candidate. This is simulated technician feedback; unverified operator notes are never training labels.

The source-selected hybrid is provisional and constrained by the project's hybrid research design. RF-only and temporal-only ablations are reported. A competitive RF-only result means the benefit of adding temporal modeling has not been established for these data; it is not hidden by the component selection.

## Reproduction and implementation

`acquire_source.py` obtains the pinned full source archive. `scripts/02_prepare_data.py` and `03_make_splits_windows.py` produce canonical observations and disjoint day assignments. `clean_data_check.py` bounds and verifies an uncached complete target archive in a fresh environment. `prepare_experiment.py --fresh-rebuild` verifies raw manifests, split records, semantic normalization and new serialized pack hashes. Historical model weights were unavailable in this workspace; no historical bundle was fabricated. New hashes are frozen for all three new runs.

`train_architectures.py --device cpu --resume` verifies or trains the three matched configurations. `run_learning.py` executes gated adaptation and reference-feedback training. `evaluate_pipeline.py` freezes selection from source validation before descriptive later-source/retrospective-target scoring, records real error examples, paired changes, threshold tradeoffs and warm CPU request times. `check_pipeline.py` exercises model identities, reloads, causality, preserved m3-v1, runtime controls and the actual Streamlit interface. `export_browser.py` exports the real fitted trees and network weights. `browser_parity.mjs` checks 900 prepared and900 raw-window predictions against Python.

The portable CPU dependencies are in the root `requirements.txt`; exact installed transitive packages are in `environment-lock.txt`. The historical Windows CUDA alternative is retained in `requirements-cuda.txt`. The actual new environment is Linux Python3.11.17, not the user's earlier Windows3.11.9 GPU environment. No hardware cost is inferred from the absence of paid API calls.

The real Windows bounded data check passed. A fresh target download from this cloud workspace returned a195-byte “Site Unavailable” page and was rejected by its pinned checksum, not treated as a successful data download. That failed attempt is retained. The actual full929MB source archive was freshly downloaded and matched its published snapshot hash. Original m3-v1 files match their Git blobs; Windows CRLF differences are accounted for without changing the originals.

## Interface controls

`app.py` is the Streamlit backend interface. The hosted HTML/JS interface runs the actual exported RF and TCN locally. Both expose recorded windows, canonical CSV input, architecture/stage choice, branch probabilities, review reasons, real comparisons and audit export. Uploaded data and every target setting require review. Recorded source cases are still research predictions, not independent validation on new equipment. No interface actuates an AHU.

The runtime policy is **m4-research-review-v2**: 15 consecutive one-minute samples, finite required channels, occupied/fan-on mode, temperature bounds, fraction roundoff tolerance1e-6, class/season scope, provisional confidence.90, branch disagreement and any sample beyond6 source standard deviations. Version2 adds mandatory review of experimental feedback candidates after actual error analysis found more fault→normal errors despite passed source-retention checks. The initial model remains the default; source retention is not operational approval. Version1 remains traceable in Git. It is a new policy, not an unacknowledged reuse of the original60-sample/.80/six-of-sixty policy. Rule fixtures establish behavior, not that these thresholds are clinically or operationally safe. Float32 inverse-normalized demo windows preserve their approximate values; original-pack scores use the frozen arrays.

Operator reviews are stored separately, require a label before verification, and never silently update weights. Browser reviews are local to that browser and exportable; the Python backend performs the controlled learning experiment. Target reference feedback is explicitly simulated.

## Evidence and publication

Actual results and configuration hashes are in `evidence/`. `pipeline_status.json` is written only after backend, numerical browser parity and interactive browser checks pass. Warm browser timing includes real inference, policy, DOM updates and two animation frames; it excludes cold downloads and page startup. CPU timings are a separate server measurement and exclude network/browser rendering.

The original repository's GitHub integration is read-only in this session and CLI pushing has no credentials. This limitation is separate from implementation completion. The release includes Git history and runnable source so the new commit remains reviewable; instructor access to a newly published branch must be resolved before the final document is submitted. No assignment or message to the professor has been sent.
