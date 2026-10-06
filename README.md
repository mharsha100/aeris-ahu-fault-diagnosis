# AERIS — dataset access warm-up

This folder supports the EGN 6216 Module 7 data collection warm-up. `playground.ipynb` retrieves both planned AHU sources: real LBNL-2019-2 target CSVs and selected simulated G36-1wk source runs. The data files are cached under `data/`, which is ignored by Git. No credentials or private operational data are needed.

## Source and rights

- Source dataset: Ghalamsiah et al., *Labeled Datasets for Air Handling Units Operating in Faulted and Fault-free States*, Figshare article 29297999, DOI [10.6084/m9.figshare.29297999](https://doi.org/10.6084/m9.figshare.29297999). The notebook requests the live article metadata and prints the retrieved version and date.
- Target dataset: Lin and Mitchell (2019), *Data Sets for Evaluation of Building Fault Detection and Diagnostics Algorithms*, [OEDI DOI 10.25984/1824861](https://doi.org/10.25984/1824861), last updated 2021-10-15. The real FLEXLAB `SZVAV.csv` and `SZCAV.csv` files correspond to the LBNL-2019-2 target named in the project blueprint.
- Licenses: **CC BY 4.0** for the Figshare dataset and **CC BY 4.0** for the OEDI release according to their respective dataset records. Cite both releases when reusing data. The articles may have different terms.
- Scope: the warm-up audits both target CSVs and selected G36-1wk source runs. Full source/target channel, unit and seasonal mapping remains future Milestone 3 work.
- Storage: authoritative files are on Figshare and OEDI, with downloaded files cached locally under `data/`. This follows the laptop/local storage choice in Milestone 2 §2.5–2.7. The GitHub repository contains code and small, displayed notebook outputs only.

## Infrastructure decision to update

Milestone 2 §2.4 and §2.7 planned a local Python/Streamlit prototype without a container. Module 7 now states that the **final deployment will use Docker**. The notebook still runs locally for data exploration, but the project blueprint should be revised to name Docker as the eventual deployment environment. A Dockerfile and container test belong to the deployment work; they are not claimed as completed in this warm-up. The `requirements.txt` and recorded Python version provide the starting point for that environment.

## Reproduce

Use Python 3.11.9 and run from this repository folder in Anaconda Prompt:

```powershell
conda create -n aeris-data python=3.11.9 -y
conda activate aeris-data
python -m pip install -r requirements.txt
jupyter lab
```

Open `playground.ipynb` and select the `aeris-data` Python kernel. Run all cells in order while connected to the internet. The small OEDI target archive downloads first; the larger Figshare source archive may take longer. Later runs use the local cache. Review the Figshare file list and set `DATA_FILE_NAME` if automatic selection needs correction. Save the notebook after the outputs appear.

## Initial audit scope

The notebook displays publisher metadata, archive/member names, CSV dimensions and types, five actual rows, null counts, duplicate rows and numeric ranges for both settings. It shows value counts for apparent target label fields but does not treat a binary fault indicator as the fault type. Source scenario labels are inferred only from filenames for *run inventory*, and are not presented as verified row labels. A few inspected source runs cannot establish class balance or coverage of the complete source/target study.

## Note for the next milestone

Record which exact runs and seasons are selected, their case-level fault labels and channel/unit mappings. Review the G36-1wk `00_explanations.pdf` and the separate LBNL-2019-2 documentation before fixing the shared classes. Split by scenario/run before creating windows to avoid leakage.
