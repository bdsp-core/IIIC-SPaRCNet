# Data source & provenance — SPaRCNet

## Raw / source data (the canonical home)
The raw EEG segments, expert labels, and model artifacts underlying this paper are
published as the credentialed BDSP dataset:

- **bdsp.io project:** `bdsp-sparcnet` — *"SPaRCNet data: Seizures, Rhythmic and Periodic Patterns in ICU Electroencephalography"*
  → https://bdsp.io/content/bdsp-sparcnet/
- **DOI:** [10.60508/cw6j-s785](https://doi.org/10.60508/cw6j-s785)
- **Access:** credentialed (BDSP DUA); data served via the BDSP restricted S3
  access point (`bdsp-restricted-access-point`). Apply on bdsp.io, then `aws s3 sync`
  per the dataset's usage notes.

## Proximal artifacts in this repo (committed, de-identified)
`code_for_figures/Data/` holds the small "last-mile" files each figure/table is
computed from directly — model probabilities, expert consensus labels, precomputed
UMAP coordinates, and cohort tables. These are **de-identified**: subjects are
surrogate IDs (`sparcnet_subject<NNNN>`), no names/MRNs/dates. Total ~60 MB across
43 `.mat` files. See `REPRODUCE.md` for which file feeds which figure.

## Raw → derived lineage
1. **Raw EEG** (10 s, 128 Hz, 19-channel bipolar segments) + expert IIIC labels — in the `bdsp-sparcnet` dataset.
2. **SPaRCNet model** (`IIIC_train_0502.py`, `code/`) is trained on the raw data → per-segment class probabilities.
3. **Proximal artifacts** (`code_for_figures/Data/*.mat`): the model probabilities +
   expert operating points (Fig 1/2), the 2-D UMAP embedding + selected samples
   (Fig 3–6), spread/IRR summaries (Fig S3/S5/S8), and cohort split tables (Table 1).
4. **Figures/tables**: the `code_for_figures/*.m` scripts turn (3) into the published
   figures — no raw data needed at this step.

To regenerate the proximal artifacts from raw data (rather than reproduce the figures
from the committed artifacts), you need credentialed access to `bdsp-sparcnet` plus the
training/inference code in `code/` and a GPU.
