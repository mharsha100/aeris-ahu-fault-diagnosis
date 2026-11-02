# AERIS Milestone 3 data pipeline

The October 12, 2026 audit prepares 33 G36-1wk runs and the two LBNL FLEXLAB CSVs. It records real data limitations and deliberately does not train a model or release an invalid test split.

## Reproduce

The measured environment is Python 3.12.14, pandas 3.0.1 and NumPy 2.3.5 on Windows 11. From this directory, install `requirements.txt` in a Python 3.12 environment, then run:

```powershell
python -m pip install -r requirements.txt
python ingest.py --cache data --record evidence/ingestion.json
python pipeline.py --cache data --output .
```

To reuse the warm-up cache, replace `data` in both commands with its full directory path. Ingestion verifies exact archive bytes and SHA-256; absent archives are downloaded from pinned publisher URLs. The original data are not edited. The Milestone 3 run reused the existing cache; it did not test a fresh 936 MB download. The earlier executed warm-up notebook provides acquisition evidence.

`processed/` contains canonical sensor CSVs, row quality flags, and target labels in separate files. Logical label separation is not an access-control boundary. `evidence/` contains the run environment, raw/processed hashes, member manifest, full G36 file inventory, target day counts, mappings, quality findings and policy checks. No raw or processed time series need to be uploaded to GitHub.

## Evaluation restriction

The earlier warm-up displayed target labels and counts. These target files are exploratory, not a sealed holdout. G36 scenarios within a season share weather and occupancy. The 33 source files contain only three independent seasonal condition groups, with one per season. Minute records and severity variants are not independent replications. No train/validation/test partition is released, no learned transform is fitted, and no model-ready windows are constructed. The checks reporting zero role overlap concern provenance partitions only; they do not establish model generalization.

Before any new final target labels are opened, require at least five independent cases for every retained class and season (including normal), independently verified scenario/channel/mode mappings, and an untouched holdout. Also require five independent source cases per class-season in training and five different cases in source validation. These are pragmatic minimum reporting rules, not power calculations. Keep every shared-weather group and every fault episode on one side of any later split. If a condition is not met, withhold the corresponding performance claim; do not lower the minimum after seeing results.

## Review policy

`review_gate` implements 60-sample input checks. It forces Needs review for unsupported settings/modes, missing/nonfinite inputs, gaps, invalid ranges, or at least six readings outside a frozen training temperature range. A high model score cannot override these checks. The range must eventually be fitted on source training only. No fitted support exists in Milestone 3, so the real prediction route remains closed. The ten automated checks use synthetic fixtures and supplied scores, not trained-model outputs; fixture bounds are not empirical validation results.

## Source attribution and rights

- Ghalamsiah, Naghmeh (2025), *Labeled Datasets for Air Handling Units Operating in Faulted and Fault-free States*, Figshare version 3, DOI https://doi.org/10.6084/m9.figshare.29297999.v3. Dataset license: CC BY 4.0.
- Lin, Guanjing and Robin Mitchell (2019), *Data Sets for Evaluation of Building Fault Detection and Diagnostics Algorithms*, Lawrence Berkeley National Laboratory, OEDI, DOI https://doi.org/10.25984/1824861; release updated October 15, 2021. Dataset license: CC BY 4.0.
- License: https://creativecommons.org/licenses/by/4.0/. Changes in this derivative preparation include header trimming, numeric parsing, Kelvin/Fahrenheit to Celsius conversion, channel selection and separate quality/label records. No endorsement is implied.
- Mapping evidence: G36 `00_explanations.pdf` tables 1–3 and note on page 10; publisher `DomainPairs.pdf` page 2; OEDI `LBNLDataSynthesisInventory.pdf` pages 17–19.

No pretrained model, personal records, live BMS connection, paid API or automatic feedback adaptation is used.
