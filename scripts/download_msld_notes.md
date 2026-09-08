# Dataset notes (manual only)

This project does **not** ship or auto-download lesion images. Training images
are third-party data with their own licenses and terms. Place files yourself
under the layout in `docs/contracts.md`.

```
data/raw/monkeypox/         # class 1
data/raw/non_monkeypox/     # class 0
```

Allowed types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

Do not add images to git. `.gitignore` ignores raw image binaries.

## Public datasets (research use)

### MSLD / MSLD v2.0 (Mpox Skin Lesion Dataset)

- MSLD v1 (binary “Monkeypox” vs “Others”):
  https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset
- MSLD v2.0 (six clinical classes, dermatologist-reviewed):
  https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20

Cite the dataset authors if you use these images, for example:

- Ali, S. N., et al. (2022). Monkeypox Skin Lesion Detection Using Deep
  Learning Models: A Preliminary Feasibility Study. arXiv:2207.03342.
- Ali, S. N., et al. (2024). A Web-based Mpox Skin Lesion Detection System
  Using State-of-the-art Deep Learning Models Considering Racial Diversity.
  Biomedical Signal Processing and Control, 98, 106742.

### Other public sources

Search Kaggle / Mendeley / papers for “mpox skin lesion” datasets. Always read
the license, Kaggle rules, and any ethics/IRB notes before use. Do not scrape
clinical photo sites or social media.

## Manual steps (no bulk download script)

1. Create a Kaggle account, accept the dataset terms, and download the archive
   in the browser (or the official Kaggle CLI on your own machine).
2. Unpack locally **outside** git if you prefer, then copy originals only.
3. Map classes into this repo’s two folders:

   | Source label                         | Destination                |
   |--------------------------------------|----------------------------|
   | Monkeypox / Mpox                     | `data/raw/monkeypox/`      |
   | Others, Chickenpox, Measles, Cowpox, HFMD, Healthy, non-mpox | `data/raw/non_monkeypox/` |

4. Prefer **original** photos. Skip vendor-provided augmented copies and skip
   any pre-made train/val/test image folders — this repo builds splits with
   SHA-256 / pHash / `group_id` rules (`python -m ml.scripts.make_splits`).
5. If patient or case IDs exist, keep them for `group_id` so the same person
   never crosses splits. Otherwise the audit/split code uses file hashes.

If `data/raw` is empty, audit / split / train must exit non-zero. Never
synthesize training images or invent metrics.
