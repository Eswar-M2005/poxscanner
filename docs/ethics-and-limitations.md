# Ethics and limitations

This system is **not a diagnosis**, not a screening programme, and not clinical decision support for patient care. The UI and API disclaimer is mandatory: *Research and educational use only. This tool does not provide a medical diagnosis.*

Laboratory methods (e.g. PCR) remain the reference standard for mpox. A softmax score on a crop of skin is not a case definition.

**Model scores: PENDING REAL TRAINING RESULTS.** No trained accuracy, AUC, or F1 is reported. `selection.json` is `pending_training`. Placeholder metric files contain nulls, not results. Demo-mode probabilities are a hash of file bytes.

## Image-only limits (literature)

Setegn GM, Dejene BE. Explainable AI for Symptom-Based Detection of Monkeypox: a machine learning approach. *BMC Infectious Diseases*. 2025;25:419. https://doi.org/10.1186/s12879-025-10738-4

That work is symptom-based. In reviewing image-based monkeypox ML, they state that **image-based disease detection has limitations and may not be effective in many real-life situations**, and that **several imaging studies lacked explainability and interpretability**, which made it hard to understand model decisions. They argue for clinical symptoms (fever, rash, lymphadenopathy, and related features) plus XAI as a more practical complement to imaging-only pipelines.

This repository is image-based because the intended dataset is photographs, not a symptom table. That choice inherits the limitations they describe:

- No access to prodrome, lymphadenopathy, exposure, vaccination, or immune status.
- Lesions of mpox overlap visually with varicella, HSV, syphilis, HFMD, and other dermatoses; a binary `non_monkeypox` bucket does not name the alternative.
- Photos are often well-lit, centred, late-stage lesions from public or web-style sets—unlike phone photos in poor light, multiple stages, or clothed/partial views.
- A high in-domain accuracy (when eventually measured) would not imply bedside utility.

We require Grad-CAM because they (and the broader XAI literature) treat unexplained CNN scores as insufficient. Grad-CAM does **not** close the gap to their tabular XAI, which attributes predictions to named symptoms.

## Not a diagnostic device

- Do not use outputs for isolation, treatment, vaccination, or reporting.
- Do not present demo-mode (`demo_mode: true`) as model output. Demo probabilities are deterministic functions of file bytes.
- Do not advertise pending/null metrics as results. **PENDING REAL TRAINING RESULTS** — do not invent or copy numbers from other papers into this project.

## Dataset bias

A local collection is present under `data/raw/` and has been audited and split (7518 enumerated, 602 exact duplicates removed, 6916 unique, 127 groups, train/val/test 6374/272/270). **Bias is still largely unmeasured** because no trained model has been evaluated on subgroups. Expect at least:

- **Source bias:** web-scraped or Kaggle-style sets over-represent canonical lesions and under-represent early, atypical, and healed lesions.
- **Skin tone bias:** many public rash datasets skew toward lighter phototypes; error rates may be worse on darker skin if that is not tested. No phototype labels are in the split CSVs.
- **Label noise:** captions and forum tags are not PCR. Noise inflates or deflates metrics unpredictably. Filename families (MKP, HFMD, CHP, CWP, MSL, HEALTHY) suggest mixed negative conditions under one class.
- **Geographic / clade shift:** 2022+ clade IIb presentations differ from historically endemic descriptions; a model fit on one may fail on the other.
- **Duplicate leakage:** republished images across train/test fake generalization (mitigated by SHA-256 and aHash grouping; see [dataset.md](dataset.md)). Residual risk: **23** near-duplicate groups span both classes in the audit — flagged, not resolved by deletion.
- **Small held-out positives:** 52 monkeypox images in val and 52 in test. Any future sensitivity estimate will have a wide interval.

Unevaluated subgroups must be stated as unevaluated, not assumed fair. Licence, consent, and confirmation standard for the local files are incompletely documented; do not claim a representative clinical population.

## Privacy of skin photographs

Lesion images can identify people (faces, tattoos, backgrounds, metadata). Treat uploads as sensitive:

- Default **`STORE_UPLOADS=false`**: ignore client filenames; do not write images to disk; do not log pixel payloads.
- Do not ship EXIF GPS or other metadata into manifests.
- `group_id` is a split key (filename prefix or hash), not a medical record number. Do not encode names, MRNs, or dates of birth.
- Obtain licence and consent appropriate to research use before adding files under `data/raw/` or hosting the demo beyond a private lab.

## No PHI storage by default

The API is designed **not** to store protected health information. Environment:

- `STORE_UPLOADS=false` (default)
- No patient identity fields in predict request/response
- Checkpoints and metrics JSON contain model weights and aggregate scores, not images, once training exists

Enabling `STORE_UPLOADS=true` is an explicit, auditable choice and requires a retention and access policy outside this default design. This project does not implement an EHR, user accounts, or clinical audit log.

Upload validation (MIME allowlist, size cap 8 MB, decode-before-trust, max dimension 4096 px) reduces accidental execution of non-images; it is not a substitute for a clinical security review.

## Grad-CAM is not a causal explanation

Grad-CAM weights spatial activations by the gradient of a class score. A hotspot means “this region influenced the logit,” not:

- “this is a pathognomonic vesicle,”
- “the patient has mpox because of this pixel group,”
- a substitute for SHAP/LIME on clinical features.

Known failure modes: highlighting bandages, nails, text, rulers, or background contrast; missing the true lesion; looking locally coherent while the decision is still a dataset artefact. Demo mode may show a **placeholder** heatmap; that is not a CAM.

Do not use overlays in patient-facing materials as proof of correctness.

## Additional limits

- **Binary task:** cannot rank differential diagnoses.
- **Single image:** no temporal course, no multi-site exam.
- **224×224 crop:** discards resolution that dermoscopy or clinical photography might use.
- **No prospective trial:** even after metrics exist, they will be retrospective on a held-out split, not clinical validation.
- **Transfer learning is implemented, not reported:** freeze-then-last-block fine-tune exists in `ml/src/train.py`; scores remain PENDING REAL TRAINING RESULTS.
- **Symptom models are a different task:** do not quote Setegn & Dejene’s 89.3% LGBM accuracy as this project’s performance.

Responsible next steps after training completes: report sensitivity/specificity with sample sizes (and intervals), slice by filename family or skin tone if labels allow, keep demo vs trained inference distinct, review cross-class near-duplicates, and keep the non-diagnostic framing.
