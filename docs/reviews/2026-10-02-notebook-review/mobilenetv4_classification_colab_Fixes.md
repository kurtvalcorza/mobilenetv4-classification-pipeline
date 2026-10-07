# mobilenetv4_classification_colab — review fixes (2026-10-02 review, fixed 2026-10-07)

Fixes for the Notebook Review Framework v1 findings (prefix `MNV`, review PR #9) on top of the fleet-sweep fixes of
2026-10-05 (`0ccaf40`, record in `../2026-10-05-fleet-sweep/`). Every change is made in the generator
(`tools/build_notebook.py`, `tools/notebook_template.py`), the package (`src/mobilenetv4_classification_pipeline/pipeline.py`)
or the validator, and the notebook is regenerated. STATUS and the release labels are unchanged.
**Readiness: Verification pending** (no hosted run of the regenerated blob yet).

## MNV-B1 root cause (from code)

The old default could not learn, for four reasons that compound:

1. `timm.create_model(..., num_classes=2)` leaves the new 2-way head at timm's random initialisation on top of
   1280-d pre-logit features. The first loss was therefore 2.4–5.0 instead of ln 2 = 0.693 (the review's "training loss
   4.018 … mean confidence 0.91"): the untrained head was already confidently wrong.
2. `AdamW(model.parameters(), lr=1e-4)` for one epoch of 7 steps (26 images, batch 4) moves each weight by at most
   ~7 × 10⁻⁴, which is small against head weights of order 0.03; the head cannot leave its random start in 7 steps.
3. Every parameter trained with BatchNorm in train mode on batches of 4, and the batches were taken in stored order
   (13 frogs, then 13 trucks), so each batch was single-class and the BatchNorm statistics were pushed per class.
4. Nothing was seeded (head init, order), so successive runs scattered between 0.17 and 0.83 on 6 held-out images.

The fix in `pipeline.fit`: `trainable="head"` by default (the backbone frozen, the whole network in eval mode so the
pretrained BatchNorm statistics are kept), `head_init="zeros"` (the first loss is exactly ln(classes)), a seeded shuffled
batch order and seeded initialisation (`seed=42`), `lr=1e-3`, `epochs=8`, `batch_size=8`, and the recipe (method,
optimizer, lr, epochs, steps, batch size, seed, head init, BatchNorm mode, trainable/frozen/total parameter counts)
returned in the metadata and written to `model-config.json`. The notebook's default dataset grows from 16 to 100
images per class (after duplicate removal), giving 160 training / 40 held-out images and 160 optimiser steps.

**What a hosted run must show (the stand-in cannot):** on the real pinned weights, `final_train_loss` in Section 8 well
below 0.693, held-out accuracy of the reloaded model above the 50 % majority baseline (and near the zero-shot
reference, which the review measured at 6/6 on the previous split), the Wilson interval printed, `VERDICT:
above-majority-baseline`, and — for the three-seed half of the acceptance check — `EXTRA_SEEDS = 2` repeated with every
seed above the baseline.

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| MNV-B1 (default fine-tune does not learn; `verdict: success` hard-coded) | Partly fixed — root-caused and corrected in code; learning on the real weights needs the hosted run | Root cause above. `pipeline.fit` rewritten (head-only default, zero head, seeded shuffle, frozen BN, recipe recorded); Section 8 exposes `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `TRAINABLE`, `SEED`, keeps 100 images per class (160 / 40 split), prints trainable/frozen counts, the recipe, every epoch and `loss_fell_below_chance`; Section 9 computes the verdict from the metrics (`compute_verdict`: above / at / below the majority baseline, and against the zero-shot reference), prints a 95 % Wilson interval and "one image = 2.5 points", and `EXTRA_SEEDS` repeats the fit for a seed spread. Expected-result notes after Sections 8 and 9 say what to expect and how to read a lift at or below zero. `grep "'verdict': 'success'"` on the notebook returns nothing. | `src/.../pipeline.py` (`fit`), Sections 8, 9, 10; Interpretation | `test_mnv_b1_default_recipe_moves_the_head_below_chance_loss_on_a_stand_in` (fit on a randomly initialised, BN-calibrated network: first-epoch loss at chance, loss falls, old recipe starts confidently wrong — **stand-in, not pretrained evidence**), `test_mnv_b1_verdict_is_computed_not_hard_coded`, `test_mnv_b1_section_8_uses_the_new_recipe_and_prints_counts`, `test_mnv_b1_prose_tells_the_learner_what_to_expect_and_how_to_read_a_zero_lift` |
| MNV-M1 (Run all needs a restart) | Fixed (sweep `0ccaf40`) — hosted confirmation pending | Isolated uv runtime; `grep "Restart the runtime"` returns nothing. | Section 1 | sweep `test_swp_r_*`; `test_mnv_m1_no_restart_and_every_cell_parses` |
| MNV-M2 (only a majority baseline; the pretrained model already solves the task) | Fixed — hosted confirmation pending | Section 9 scores the untouched pretrained `pipe` on the same held-out split through the ImageNet-1k groups (`frog` → 30–32, `truck` → 555, 569, 675, 717, 864, 867; summed softmax per group), prints majority, zero-shot and fine-tuned side by side with the verdict, and the Interpretation says the zero-shot reference is expected to be highest or tied and what the fine-tune demonstrates instead; on classes without a mapping the reference is reported as unavailable. | Section 9; Interpretation and limits; Glossary | `test_mnv_m2_zero_shot_reference_reads_the_pretrained_scores_through_groups` |
| MNV-M3 (head-only described, all trained; unseeded; unrecorded) | Fixed | Prose and code agree (head-only default; `TRAINABLE = "all"` is the labelled alternative); trainable/frozen counts printed; head init and shuffled order seeded from `SEED` (Python's `random`, used by timm's crop, is seeded inside `fit` and restored after); recipe in `result.json["fine_tuning"]["recipe"]`, the evaluation report and `model-config.json`. | `pipeline.py`; Sections 8, 10 | `test_mnv_m3_two_fits_are_identical_and_the_backbone_is_untouched` (identical scores for the same seed, different for another; only `classifier.*` tensors move), `test_mnv_m3_fit_refuses_bad_arguments` |
| MNV-M4 (guided layer absent) | Fixed (sweep `0ccaf40` + this pass) | Sweep guided layer; this pass adds expected-result notes after Sections 8 and 9, a prediction about the loss start and the three-way ordering, and a Change-one-thing activity (the old recipe). | Sections 8, 9 | sweep `test_swp_g_*`; `test_mnv_b1_prose_…` |
| MNV-m1 (duplicate pairs; independence unstated) | Fixed | Pixel-identical images are grouped by RGB digest before any split (first kept, the rest counted in `skipped_members['pixel_duplicate']`); the prose names the darkened copies and the independence assumption; the printed dict carries `split_assumption`. | Section 8 | `test_mnv_m1_pixel_identical_copies_are_grouped_before_the_split` (every image shipped twice: 0 held-out copies in training) |
| MNV-m2 (fallback `NameError`; digest mismatch swallowed) | Fixed | The `try` covers only the download; the SHA-256 check is outside it and fails closed with the expected/actual digests and byte count; the network fallback is labelled in `dataset_source`, `dataset_fallback_used` in the evaluation report and `result.json`, and disables the zero-shot reference. (The `NameError` itself was fixed by the sweep.) | Section 8, 9, 10; Troubleshooting | `test_mnv_m2_tampered_sample_archive_fails_closed`, `test_mnv_m2_offline_fallback_is_labelled` |
| MNV-m3 (BYOD branches) | Partly fixed | Sweep: named decode errors, empty upload, root-level/val-only reporting. This pass: a new-data inference step after reload (the Section 4 image, outside both splits). **Not done:** dataset images are still not passed through `validate_inputs` into a dataset manifest (the single-image validator's ceilings would need a batch form; left for the maintainer). | Section 9 | `test_mnv_m3_new_data_inference_and_held_out_table`; sweep `test_swp_b_*` |
| MNV-m4 (prerequisites and release documents contradict) | Partly fixed | Prerequisites name the dataset download (~986 KB, Hub, pinned commit, digest-refused) and no longer say "nothing is downloaded". **Not changed:** `docs/release-verification.md`, `README.md`, `STATUS.md` and the registry status are release records and labels (maintainer decision; the brief forbids status changes). | Prerequisites | `test_mnv_m4_prerequisites_name_the_dataset_download` |
| MNV-m5 (reload never compared) | Fixed | Section 9 compares the reloaded model's scores with the in-memory fine-tuned model on every held-out image, prints the maximum difference with the 1e-5 tolerance and raises when exceeded (contract integrity). | Section 9 | `test_mnv_m5_reload_equivalence_is_checked_with_a_tolerance` |
| MNV-S1 (spec 2.2) | Not done | Maintainer decision. | — | — |
| MNV-S2 (show held-out images) | Fixed (table) | The first twelve held-out images are listed with true label, predicted label, score and ok/WRONG. | Section 9 | `test_mnv_m3_new_data_inference_and_held_out_table` |
| MNV-S3 (Wilson interval) | Fixed | `wilson_interval` printed and exported; "one image = 2.5 points". | Section 9 | `test_mnv_s3_wilson_interval` |
| MNV-S4 (`mode='RGB'`) | Fixed | Dropped in both `Image.fromarray` calls. | Sections 4, 8 | `test_mnv_s4_no_deprecated_fromarray_mode` |

## User-visible changes

- `pipeline.fit` defaults changed: `epochs=8`, `batch_size=8`, `learning_rate=1e-3`, new keywords `trainable="head"`, `seed=42`, `shuffle=True`, `head_init="zeros"`; the returned metadata and `model-config.json` gain `fine_tuning`. Full fine-tuning remains available with `trainable="all"`.
- Section 8: new form fields `SEED`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `TRAINABLE`; `SUBSET_PER_CLASS` 16 → 100 (160 training / 40 held-out images; a few minutes on CPU instead of seconds); duplicate removal; the digest mismatch now stops the notebook; the fallback is labelled.
- Section 9: reload-equivalence check (raises on mismatch), zero-shot reference, Wilson interval, computed verdict, held-out table, new-data inference, `EXTRA_SEEDS`; the evaluation report gains `zero_shot_comparison`, `accuracy_wilson_95`, `zero_shot_reference_accuracy`, `reload_max_abs_score_diff`, `fine_tuning`, `dataset_fallback_used` (and `seed_spread` when requested); `verdict` is now one of `above-majority-baseline`, `at-majority-baseline …`, `below-majority-baseline …`.
- `result.json["fine_tuning"]` gains `recipe`, `dataset_source`, `dataset_fallback_used`, `skipped_members`, `seconds`.

## Verification (offline; not clean-runtime evidence)

- **Real input:** the pinned weights cannot be fetched here (Hub unreachable), so no cell ran on the real model and the review's `run_probes.py` was not re-run. All numbers in the prose are the review's recorded measurements, named as such.
- **Stand-in:** `tests/test_review_fixes.py` runs `pipeline.fit` on a randomly initialised MobileNetV4-Conv-Small whose BatchNorm statistics were calibrated on noise (timm's `create_model` patched; the fake snapshot passes the identity checks): the zero head starts at chance loss and the loss falls under the new recipe, the old recipe starts confidently wrong, two fits with one seed are identical, only `classifier.*` tensors move. Random features are not discriminative, so held-out accuracy is **not** asserted — that is the hosted run's job. Section 8 is executed on in-memory archives (duplicates, tampered digest, offline); the Section 9 helpers (verdict, Wilson, zero-shot grouping) on fixed inputs.
- Commands (CI's own: torch CPU, timm 1.0.29): `ruff check src tests tools` clean; `pytest` 49 passed → 65 passed; `tools/validate_release_assets.py` PASS (two markers updated); `tools/build_notebook.py --check` OK.

## Remaining gates

- A hosted one-pass Run all on a fresh runtime of the regenerated blob (MNV-M1 acceptance), recorded with the blob id; its Section 8 `final_train_loss` and Section 9 verdict/Wilson line checked as listed above (MNV-B1 acceptance, hosted half), plus a run with `EXTRA_SEEDS = 2` for the three-seed spread.
- The REL12 BYOD runs (image and dataset).
- Maintainer decisions: the four release documents' agreement and the registry status (MNV-m4), a dataset input manifest for BYOD datasets (MNV-m3), spec declaration 2.0 → 2.2 (MNV-S1).
