# Dataset

`data/raw/monkeypox/` and `data/raw/non_monkeypox/` are populated locally (not committed — `data/raw/**/*.{jpg,jpeg,png,webp,bmp}` is gitignored; only `.gitkeep` placeholders are tracked). Audit, split generation, and training still **must exit non-zero** with a clear message if `data/raw` is empty on a fresh checkout. **Do not synthesize training images.**

Source images were copied from a local `datsets/Monkeypox/` and `datsets/non-monkeypox/` folder into the contract layout below; that source folder is not itself consumed by the pipeline.

## Current audit snapshot (local, not committed)

From `ml/artifacts/metrics/audit_report.txt` / `audit_report.json` (scanned 2026-08-28; regenerate with `python -m ml.src.audit`):

**Audit status: `issues`** — not a clean pass. The auditor flagged **23 cross-class near-duplicate groups** (aHash Hamming ≤ 5 spanning `monkeypox` and `non_monkeypox`). Cross-class *exact* SHA-256 groups: **0**. Those 23 groups are a real labeling / look-alike-lesion concern and need manual review; they are **not** hidden by a green status.

| Class | Files found | Valid | Corrupt |
| --- | --- | --- | --- |
| `monkeypox` | 2828 | 2828 | 0 |
| `non_monkeypox` | 4690 | 4690 | 0 |

- Total enumerated / valid: **7,518** (all JPEG, none corrupt).
- Imbalance ratio (majority/minority): **1.66** (1.6584 in the audit report).
- **602** exact SHA-256 duplicate groups; **602** extra copies removed before splitting (`7518 − 602 = 6916`).
- **849** perceptual near-duplicate groups (aHash Hamming ≤ 5) — mostly augmented variants of the same lesion photo (filenames like `MKP_01_01_1.jpg` … `_13.jpg` plus `_ORIGINAL.jpg`).
- Split grouping itself is by filename-derived patient/session id, not by the near-duplicate cluster alone, so the 23 cross-class near-dups did not by themselves move images across train/val/test. They remain an audit **issue**, not a split-leakage finding.

From `data/splits/split_meta.json` and the split CSVs (regenerate with `python -m ml.scripts.make_splits`):

| Split | monkeypox | non_monkeypox | total | groups |
| --- | --- | --- | --- | --- |
| train | 2502 | 3872 | 6374 | 88 |
| val | 52 | 220 | 272 | 19 |
| test | 52 | 218 | 270 | 20 |

- **6,916** unique images kept (`6374 + 272 + 270 = 6916`). CSV columns: `path,label,split,sha256,width,height,group_id`.
- Grouped into **127** patient/session groups (via filename prefix, e.g. `MKP_17_*` or `HFMD_34_*`; `88 + 19 + 20 = 127`) so all augmented copies of one lesion stay in a single split.
- CSV-column check (no image re-hash): **zero** `sha256` overlap across train/val/test; **zero** `group_id` overlap across train/val/test.
- Because each group holds ~14–20 images and there are only 127 groups total, the group-level 70/15/15 split does not translate into an even 70/15/15 **image** count per class — val/test are intentionally smaller than a naive image-level split would give. This trades split size for leakage safety and is the correct behavior per the split policy below; treat val/test metrics accordingly (wider confidence intervals, especially for the monkeypox class at n=52).

## Expected structure

```
data/raw/monkeypox/         # positive class (index 1)
data/raw/non_monkeypox/     # negative class (index 0)
data/processed/             # optional cached tensors / manifests
data/splits/
  train.csv
  val.csv
  test.csv
  split_meta.json
```

Allowed image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

Class encoding:

```
LABEL_TO_INDEX = {"non_monkeypox": 0, "monkeypox": 1}
INDEX_TO_LABEL = {0: "non_monkeypox", 1: "monkeypox"}
DISPLAY        = {"non_monkeypox": "Non-Monkeypox", "monkeypox": "Monkeypox"}
```

Positive class for recall / sensitivity / PR-AUC: **monkeypox (1)**.

## Split manifests

CSV columns (required):

```
path,label,split,sha256,width,height,group_id
```

| Column | Rule |
| --- | --- |
| `path` | Repo-relative POSIX path |
| `label` | `monkeypox` \| `non_monkeypox` |
| `split` | `train` \| `val` \| `test` |
| `sha256` | Content hash of the file |
| `width`, `height` | Pixel size after decode |
| `group_id` | Patient/group id if known; otherwise the file SHA-256 |

Generate with `python -m ml.scripts.make_splits` after a successful audit.

## Leakage rules

1. Deduplicate exact SHA-256 matches (keep one file).
2. If near-duplicates exist (perceptual hash Hamming distance ≤ 5), assign the **entire group** to one split only.
3. If a `group_id` / patient metadata file exists, split **by group**, not by image.
4. Otherwise: **stratified 70 / 15 / 15** by label (train / val / test).
5. The same SHA-256 or pHash group must never appear in two splits.

These rules exist because lesion datasets often republish the same photograph (mirrors, crops, web scrapes). Random image-level splits inflate accuracy and make val/test gaps meaningless.

Default seed: `42`.

## Reproducing on a fresh checkout

1. Obtain images you are legally allowed to use for research (licence, consent, and redistribution terms).
2. Copy them into `data/raw/monkeypox/` or `data/raw/non_monkeypox/` only. Do not train from ad-hoc folders.
3. Run `python -m ml.src.audit`. It reports counts, hashes, and problems (unreadable files, mixed labels, duplicates).
4. Run `python -m ml.scripts.make_splits` to write `data/splits/*`.
5. Then train. `python -m ml.src.train` must refuse if splits are missing or empty.

Until those steps succeed, metrics JSON may exist as **null placeholders** (`n_samples: 0`, `notes: "Pending training — no dataset."`). That is not an experimental result.

## Provenance notes

Source: local `datsets/Monkeypox/` and `datsets/non-monkeypox/` folders (filenames such as `MKP_17_02_9.jpg`, `HFMD_46_01_2.jpg`, `HEALTHY_84_01_7.jpg` suggest multiple non-monkeypox conditions — hand/foot/mouth disease, healthy skin, chickenpox — grouped under one negative class, plus per-lesion augmented crops). Licence, consent, collection dates, geography, device/camera mix, and PCR-confirmed vs. clinician-assigned vs. web-scraped labeling are **not currently documented** and should be tracked down before any claim beyond "research/educational demo" is made. Web-scraped "monkeypox vs other rash" sets are biased toward typical, well-lit lesions and under-represent early, atypical, and dark-skin presentations.

This project does not ship patient identifiers. `group_id` is an opaque split key, not a medical record number.
