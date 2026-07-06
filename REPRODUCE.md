# Reproduce — SPaRCNet (Jing et al., Neurology 2023)

Every main-text figure/table regenerates from **committed, de-identified proximal
data** under `code_for_figures/Data/` (surrogate subject IDs, e.g.
`sparcnet_subject1026`; no PHI). No raw data download or model retraining needed.

## One command
```matlab
cd code_for_figures
run_all           % see run_all.m — runs every script below, writes the PNGs
```
Or run any single script (each `cd`-independent from `code_for_figures/`):

| Paper item | Script | Input (committed) | Output |
|---|---|---|---|
| Table 1 (cohort splits) | `Table1_Splits.m` | `Data/Table1/patient_demo.mat`, `Data/Table1/dataset{1..4}.mat` | stdout |
| Figure 1 (ROC) | `Figure1_ROC.m` | `Data/Figure1/figure1_input.mat` | `Fig1.png` |
| Figure 2 (PR) | `Figure2_PR.m` | `Data/Figure1/figure1_input.mat` | `Fig2.png` |
| Figure 3 (UMAP "starfish") | `Figure3_UMAPs.m` | `Data/Figure3/figure3_input.mat`, `Data/Figure3/samples.mat` | `Fig3.png` |
| Figure 4 (SZ/LPD samples) | `Figure4_samples_SZ_LPD.m` | `Data/Figure3/samples.mat` | `Fig4.png` |
| Figure 5 (GPD/LRDA samples) | `Figure5_samples_GPD_LRDA.m` | `Data/Figure3/samples.mat` | `Fig5.png` |
| Figure 6 (GRDA/Other samples) | `Figure6_samples_GRDA_Other.m` | `Data/Figure3/samples.mat` | `Fig6.png` |
| Figure S2 (flowchart) | `FigureS2_Flowchart.m` | `Data/FigureS2/*.mat` | figure |
| Figure S3 (SP spread) | `FigureS3_SPspread.m` | `Data/FigureS3/FigureS3_input.mat` | `FigS3.png` |
| Figure S5 (UMAP spread) | `FigureS5_UMAPspread.m` | `Data/FigureS5/FigureS5_input.mat` | `FigS5.png` |
| Figure S8 (IRR) | `FigureS8_IRR.m` | `Data/FigureS8/FigureS8_input.mat` | `FigS8.png` |

Committed reference outputs (`Fig1.png` … `FigS8.png`) let you diff your regenerated
figures against the published versions.

## Requirements
- **MATLAB** R2019b+ (developed on R2020a). Toolboxes: Statistics and Machine
  Learning Toolbox. **Figure 3/S5 use UMAP** — the
  [`umap`](https://www.mathworks.com/matlabcentral/fileexchange/71902) File Exchange
  package must be on the MATLAB path (the 2-D coordinates are precomputed in
  `figure3_input.mat`, so UMAP is only needed if you recompute the embedding).

## Provenance
The committed `.mat` are the **proximal artifacts** (model outputs + expert labels +
2-D embeddings) derived from the raw SPaRCNet EEG dataset. See `DATA_SOURCE.md` for
the raw-data home and the raw→derived lineage.
