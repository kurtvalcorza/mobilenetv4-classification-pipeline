# MobileNetV4-Conv-Small Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/mobilenetv4-classification-pipeline`  
**Notebook:** `tutorials/mobilenetv4_classification_colab.ipynb`  
**Reviewed commit:** `fcb47663cd1480fdc12290c92cbc638007b7b2a6` (`main`, confirmed with `gh api repos/kurtvalcorza/mobilenetv4-classification-pipeline/commits/main`)  
**Notebook Git blob:** `484cc1023f6e577a9c1b9db48a0dd9729c3a0e8f`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-14 (commit `258fecf`); the notebook has not changed since.  
**Finding prefix:** `MNV`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The inference half of the notebook is sound. It carries `pipeline.py` verbatim, asserts the inline manifest against the module identity, stages and re-hashes the pinned `timm/mobilenetv4_conv_small.e2400_r224_in1k` snapshot, classifies a deterministic synthetic gradient, writes an input manifest with a recorded rejection, and says plainly that the gradient has no ground truth and that softmax scores are uncalibrated. Those cells ran identically on the hosted T4 and on local CPU.

The fine-tuning half, which is what makes this an `E2E` notebook, does not demonstrate what it claims. It is an older generator than the ConvNeXt V2 / CvT / EfficientNet / MaxViT notebooks (relay rows 15, 17, 23, 46): it has no zero-shot or untrained-head baseline, no `FREEZE_BACKBONE` control and no unseen split. The sibling defects therefore show up differently here:

- **Zero-shot at ceiling → worse.** The pretrained ImageNet labels already separate the `frog`/`truck` split perfectly (6/6 held-out, 26/26 train; probe P5), but the notebook never measures that. Its own fine-tune scores **1/6 = 16.7 % against a 50 % majority baseline** on the recorded T4 run and in this review's clean CPU run, with mean confidence 0.91, and the report still says `"verdict": "success"` (MNV-B1, MNV-M2).
- **`FREEZE_BACKBONE` rerun problem: not present.** There is no such control, and `fit` is a classmethod that builds a fresh model from the verified snapshot on every call, so re-running cells 17–19 retrains from the base (P10).
- **Duplicate-image leak: latent, not active at the default.** The archive holds 100 pixel-identical original/darkened pairs and the split ignores them, but the default `SUBSET_PER_CLASS = 16` happens to put no held-out image's copy in training (0/6). At 50, 100 and 200 per class the leak is 3/20, 7/40 and 30/80 (MNV-m1).

What stands in the way of `Ready for intended use`:

1. **The central fine-tuning demonstration fails and is reported as a success (MNV-B1).**
2. **No one-pass `Run all` (MNV-M1).** The recorded Kaggle run stopped at the install guard and passed only after a restart; the release record reports PASSED.
3. **The comparison omits the reference that matters (MNV-M2).** Only a majority baseline is shown; the pretrained model already solves the task.
4. **The adaptation is misdescribed and unseeded (MNV-M3).** The prose says a new head is trained; `fit` trains every parameter, with no seed on initialisation or batch order.
5. **The guided layer is largely absent (MNV-M4).**

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and PIL image handling; what a softmax over class logits is" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA used when present |
| Promised outcomes (opening cell) | pinned install; snapshot staged and digest-verified; synthetic sample validated → input manifest; classification with argmax + top-k; zero-shot evaluation report; `Cleanlab/cifar-10-subset` downloaded, digest-verified, seeded 80/20 split; bounded in-kernel fine-tune of "a new classification head"; export; fresh-boundary reload; held-out evaluation against a majority baseline; outputs + provenance; two optional BYOD branches; air-gapped fallback to synthetic stripes |
| Learning objectives | includes "execute 100% in-kernel fine-tuning on custom classes", "export and fresh-reload fine-tuned artifacts", "exercise an optional BYOD path" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried module `src/mobilenetv4_classification_pipeline/pipeline.py` @ `352ea200ad76` |

### Existing execution evidence

- `docs/release-verification.md` records one run: 2026-09-14, commit `258fecf` / blob `484cc1023f6e`, Kaggle T4, "**PASSED** — 10/10 ok code cells". That blob is the reviewed blob.
- The run's evidence (workspace `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-mobilenetv4-classification/v1/`) shows pass 1 failed in cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.1; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime…`; `restarted_after_install_cell: true`; pass 2 ran 10/10. The ledger line reads "10/10 ok (1 restart after install cell)". The record in the repository omits the restart.
- The same run's `mobilenetv4_classification_finetuned_evaluation_report.json`: accuracy 0.1667 (1/6), majority baseline 0.5, lift −0.3333, mean confidence 0.91, verdict `success`.
- No Colab run is recorded. There is no `docs/execution-evidence/` directory.

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=6`), build venv `dimer-next16` (Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.0, timm 1.0.29, numpy 2.5.3, pillow 11.3.0, safetensors 0.8.0, huggingface-hub 0.36.2 — the notebook's `PINS`). The install cell was skipped through the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; nothing was installed. `HF_HOME` pointed at an empty scratch directory, so the snapshot and the dataset were fetched fresh. `google.colab.files.upload` was a shim returning prepared bytes. **This is not a Colab run.**
- **Clean default, run twice from empty scratch directories:** both 10/10 cells. Attempt 1 (host shared with other sessions' probes; 568.6 s of cell time) gave fine-tuned held-out 3/6 = 0.5; attempt 2 (69.2 s) gave **1/6 = 0.1667**, the same as the T4 record. Synthetic-gradient top-5 identical to T4 (`spotlight, spot` 0.0322 …). Attempt 1 was later stopped by the probe timeout during P4, so its namespace was rebuilt in attempt 2; its P2 record is kept in `results.json`.
- Probe results are in the attached `mobilenetv4_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).

## 2. Separate judgments

- **Technical correctness:** the inference path, snapshot verification and artifact export are correct; the reloaded artifact reproduces the in-memory model exactly (max |Δscore| 0.0 on 6 images, P3), though the notebook never checks this (MNV-m5). Defects: restart-dependent install (MNV-M1); the training loop trains all parameters with no shuffle and no seed (MNV-M3); the dataset fallback crashes with `NameError` and swallows a digest mismatch (MNV-m2).
- **Scientific validity:** the fine-tuned model is no better than, and in the recorded run worse than, the majority baseline on a 6-image held-out split, while the untested pretrained model solves it (MNV-B1, MNV-M2). The split ignores duplicate pairs and the notebook does not state the independence assumption (MNV-m1).
- **Promise fulfilment:** inference, validation, export and reload promises are met. "Fine-tuning of a new classification head" is not what runs (MNV-M3), and the fine-tune does not adapt the model to the task (MNV-B1). The air-gapped fallback promise is false (MNV-m2).
- **Learner experience:** the inference sections are clear and candid; the fine-tuning sections give no expected-result note, no interpretation of the evaluation, and present `verdict: success` for a failed adaptation (MNV-B1, MNV-M4).
- **Spec conformance:**
  - Unresolved applicable MUSTs: RUN1, RUN10, ENV6, REL2 (MNV-M1); DAT5 and FT2 in substance (MNV-B1); FT3, FT5, FT6, ENV7, OUT8 (MNV-M3); SPL3 (MNV-m1); DAT19 (MNV-m3); VER5 (MNV-m5); REL10 (MNV-m4); REL12 BYOD evidence absent from the release record.
  - SHOULD deviations: EVAL10, EVAL11 (MNV-M2); SPL10 (MNV-m1); GDL1–GDL4, GDL6–GDL14, UX4, UX5, UX8 (MNV-M4); UX10 (MNV-m2, MNV-m3); VER4 (MNV-m5).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 10/10 | Section 1 says the cell "stops with a restart instruction" | **Not met** (MNV-M1) |
| Pinned snapshot staged and digest-verified | cell 7 | 3 files fetched, verified, `source: local-snapshot` (T4 and CPU) | printed dicts | Met |
| Synthetic sample → input manifest with a recorded rejection | cells 9, 11 | manifest `accepted` + `oversized-probe` rejected | "Look for a dictionary…" | Met |
| Argmax + uncalibrated top-5 | cell 13 | top-5 list, low spread on the gradient, as the prose predicts | correct | Met |
| Zero-shot evaluation report `not-measurable` without ground truth | cell 15 | `not-measurable` + `needs` | correct | Met |
| Dataset digest-verified; air-gapped fallback | cell 17 | digest verified when online; offline or tampered → warning then `NameError: SAMPLE_HEIGHT` | prose: "gracefully falls back" | **Not met** (MNV-m2) |
| Fine-tune "a new classification head" | `pipe.fit` → `AdamW(model.parameters())` | all parameters trained, 1 epoch, 7 steps, class-sorted batches, unseeded | prose says head adaptation | **Not met** (MNV-M3) |
| Fine-tuned model evaluated against a majority baseline | cell 19 | T4 and CPU: 1/6 vs 0.5, verdict `success`; 12 fits range 0.17–0.83 | no interpretation; `success` | **Misleading** (MNV-B1) |
| Fresh-boundary reload | cell 19 `from_pretrained(weights_dir=ft_output_dir)` | loads and evaluates; never compared to the in-memory model (probe: identical) | "verify artifact integrity" | Partly met (MNV-m5) |
| Outputs + provenance | cell 21 | 7 files + 2 artifact files; no fine-tune hyperparameters or seed | — | Partly met (MNV-M3) |
| BYOD image | cells 9–21 with `USE_BYOD=True` | photo + GT 867 → `sample-sanity` (top-1 0, top-5 1); text-as-png → raw `UnidentifiedImageError`; empty upload → `StopIteration` | — | Partly met (MNV-m3) |
| BYOD dataset | cells 17–21 with `USE_BYOD_DATASET=True` | 2×6 class folders → split 10/2, fine-tune, export, reload, eval; no new-data inference; several silent rewrites | — | Partly met (MNV-m3) |
| Objective: "execute 100% in-kernel fine-tuning on custom classes" | cell 17 | runs; learns nothing measurable | — | Exercised, not demonstrated (MNV-B1) |
| Objective: read argmax/uncalibrated scores correctly | cells 12–13 prose | — | prose correct; no learner activity | Explained, not exercised (MNV-M4) |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection of all 23 cells (10 code; carried module 513 lines, digest matches) | Inference sections are clear and honest. Prerequisites say "nothing is downloaded" although cell 17 downloads a dataset (MNV-m4). Fine-tuning sections give no expected result, no interpretation, and a hard-coded `success` (MNV-B1); the guided layer is mostly absent (MNV-M4). |
| Clean default | Documented: Kaggle T4 2026-09-14 on the reviewed blob, pass 1 failed at the install guard, pass 2 10/10 after restart, 168.9 s (MNV-M1), fine-tuned 1/6. Direct: two local CPU runs from empty caches, install skipped, 10/10 cells each; fine-tuned 3/6 and 1/6; inference outputs identical to T4. No Colab run. | Runs (after a restart on hosted images); the fine-tune result is not meaningful (MNV-B1). |
| Active learning | Direct (CPU). The notebook has no Predict → Change → Observe activity. Tried: re-running cells 17–19 twice (fresh model each time, 0.667 then 0.333 — no stale state, but unseeded); `SUBSET_PER_CLASS = 50` (a plain constant, not a form field) → split 80/20, 0.65 vs 0.50; "Next experiments" item 1 (BYOD photo with a known class) → report switches to `sample-sanity` as promised. | Controls reach the computation; results swing with the unseeded run (MNV-M3) and the only documented exercise is BYOD. |
| Reuse and recovery | Direct (CPU) via the upload shim: 5 image cases, 8 dataset-zip cases; P7 network failure and tampered download. Colab upload dialog not verified (shim). | Rules enforced with clear messages for: one class, one-image class, `../` member, non-`.zip` name, out-of-range GT, 5000 px image. Raw or missing messages for: non-image file, empty upload, undecodable zip member. Silent rewrites: root-level images → class `unknown`; val-only class dropped. Offline / tampered dataset → `NameError` (MNV-m2, MNV-m3). |

## 5. Findings

### Blocker

#### MNV-B1 — The default fine-tune does not learn the task, scores at or below the majority baseline, and is reported as `success`

- **Cell/section:** Section 8 (cell 17) and Section 9 (cell 19). Generator: `tools/notebook_template.py` lines 235 (`SUBSET_PER_CLASS = 16`), 343–345 (`epochs=1`, `batch_size=4`, `learning_rate=1e-4`), 397–400 (majority baseline, hard-coded `'verdict': 'success'`); `src/mobilenetv4_classification_pipeline/pipeline.py` `fit` lines 387 (`AdamW(model.parameters())`) and 398 (unshuffled batches).
- **Observed issue:** the default run trains for one epoch of 7 AdamW steps (26 images, batch 4) and evaluates on 6 held-out images. The reported result is 1/6 correct against a 3/6 majority baseline. The evaluation report sets `"verdict": "success"` unconditionally, prints "Lift Over Baseline: −33.33%" with no comment, and the Interpretation section says successful execution proves the notebook can "perform in-kernel fine-tuning".
- **Consequence:** the central `E2E` demonstration — adapt a pretrained model to custom classes and show it on held-out data — shows an adapted model that is worse than guessing the majority class, with mean confidence 0.91, and labels that a success. A learner either concludes that fine-tuning made the model worse and the notebook calls that success, or copies `success` without reading the numbers. Both teach the wrong conclusion. Because one image is 16.7 points on a 6-image split, no run of this configuration can show a real effect.
- **Evidence:**
  - Documented: Kaggle T4, blob `484cc102`: accuracy 0.1667, baseline 0.5, lift −0.3333, mean confidence 0.91, verdict `success`; per class frog 0/3, truck 1/3; training loss 4.018 (an uninformed constant predictor on 2 balanced classes scores ln 2 = 0.693).
  - Direct (CPU, P2): clean run 1/6 (training loss 5.04); earlier clean run 3/6.
  - Direct (CPU, P4/P10): 12 fits of the same configuration ranged 0.17–0.83 (seeded as shipped 0.50 / 0.67 / 0.50; unseeded 0.67 / 0.50; reruns 0.67 / 0.33; seeded with shuffled order 0.83 / 0.50 / 0.83); training loss 2.36–5.04 in every fit.
  - Direct (CPU, P5): the pretrained model, read through ImageNet frog (30–32) and truck (555, 569, 675, 717, 864, 867) classes, scores 6/6 held-out and 26/26 train.
  - Inferred cause (not isolated): seven full-network steps at lr 1e-4 with batch-4 BatchNorm in train mode do not move a randomly initialised head past its starting loss; shuffling alone did not fix it.
- **Recommended correction:** make the default adaptation one that measurably learns, and make the evaluation able to show it. For example: train only the new head (or keep BatchNorm in eval mode) with a seeded shuffled loader and enough steps for the loss to fall well below ln 2; enlarge the held-out split (at least ~40 images) and report an interval; derive the verdict from the measured lift instead of hard-coding `success`; add an expected-result note and an interpretation that states the principal result against the baselines (including the zero-shot reference of MNV-M2). Implement in `pipeline.fit` and `tools/notebook_template.py`, regenerate, and re-run on a hosted runtime.
- **Acceptance check:** this test passes when all of the following hold.
  - On a hosted one-pass run and on three seeds, held-out accuracy of the reloaded fine-tuned model exceeds the majority baseline, and the notebook prints the interval or seed spread.
  - `grep -n "'verdict': 'success'" tutorials/mobilenetv4_classification_colab.ipynb` returns nothing; the verdict is computed from the metrics.
  - The cell after the evaluation tells the learner what result to expect and how to read a lift at or below zero.
- **Spec:** DAT5, FT2, FT7, EVAL3, EVAL6.

### Major

#### MNV-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1. Generator: `tools/build_notebook.py` `_INSTALL_GUARD` (lines 47–70) and the install-cell assembly (line 424). Also `docs/release-verification.md` Recorded executions, `tutorials/README.md` "Run-all: verified".
- **Observed issue:** the cell `pip install`s eight pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. The opening cell promises that **Run all** in a fresh runtime completes every stage with no intervention.
- **Consequence:** on a stock Kaggle or Colab image the first code cell errors and the learner must restart and run again. "PASSED — 10/10 ok code cells" rests on that restart-dependent run.
- **Evidence:** documented: Kaggle T4 run of blob `484cc102`; pass 1 stopped (`cuda-bindings 12.9.4 → 13.4.1`, `numpy 2.0.2 → 2.5.3`), `restarted_after_install_cell: true`, pass 2 10/10; the workspace ledger says "1 restart after install cell". Source: P1 `pip_install_in_kernel: true`, `uses_uv: false`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** this test passes when both hold.
  - A fresh Kaggle or Colab runtime completes every code cell in one pass with no restart, recorded with the blob id and `restarted: false`.
  - `grep -n "Restart the runtime" tutorials/mobilenetv4_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### MNV-M2 — The fine-tune is compared only with a majority baseline, although the pretrained model already solves the task

- **Cell/section:** Sections 8–9 (cells 16–19), Interpretation (cell 22). Generator: `tools/notebook_template.py` Section 9 cell (lines 362–435).
- **Observed issue:** the only reference is the majority class. The `frog`/`truck` CIFAR task maps directly onto ImageNet-1k classes the pretrained model already knows, but the notebook never scores the pretrained model on the same split. This is the siblings' zero-shot-at-ceiling problem: here the ceiling is never shown, so the learner cannot see that adaptation lost ground.
- **Consequence:** the learner cannot answer the question an adaptation experiment exists to answer — what did fine-tuning add over the starting model? — and "Next experiments" sends them to "adapt to your domain" without that comparison.
- **Evidence:** direct (CPU, P5): grouped ImageNet mapping 6/6 held-out, 26/26 train; fine-tuned 1/6 on the same split (P2) and on T4 (documented).
- **Recommended correction:** add a zero-shot reference on the held-out split (the grouped ImageNet mapping, or the pretrained features with an untrained head), print it beside the majority baseline and the fine-tuned score, and either choose a tutorial dataset whose classes are not already ImageNet classes or say explicitly that the default task is already solved zero-shot and what the fine-tune demonstrates instead.
- **Acceptance check:** the evaluation cell prints majority, zero-shot and fine-tuned held-out accuracy from the same split, and the Interpretation names which is highest and why.
- **Spec:** EVAL10, EVAL11, GDL7, GDL14.

#### MNV-M3 — The adaptation is described as head-only, trains every parameter, and is not seeded or recorded

- **Cell/section:** opening cell ("100% in-kernel classification head fine-tuning"), Section 8 prose ("replaces the 1000-class head with a new linear classifier … initializes from the verified backbone"), `pipeline.fit` (docstring "Fine-tune the classification head", line 387 `AdamW(model.parameters())`, lines 392–414 unshuffled loop), cell 21 export.
- **Observed issue:** `fit` optimises all parameters with BatchNorm in train mode, iterates the training list in stored order (13 frogs then 13 trucks, P2 `train_targets_order`), and seeds neither the head initialisation nor the batch order (`SEED = 42` seeds only the split). The trainable/frozen set is never stated. `result.json` records the history and classes but not the method, learning rate, epochs, batch size, seed or trainable-parameter count.
- **Consequence:** the learner is told one adaptation method and runs another; the same notebook gives different held-out accuracy on each run (0.17 to 0.67 across unseeded runs), and the exported artifact cannot be interpreted from its provenance.
- **Evidence:** source (P1: `fit_optimizer_all_params: true`, `fit_shuffles_training_order: false`, `fit_seeds_torch: false`); direct (P2, P4, P10 spreads above).
- **Recommended correction:** make the prose and the code agree (either freeze the backbone and say so, or describe full fine-tuning); print the trainable and frozen parameter counts; seed initialisation and a shuffled loader from the notebook's `SEED`; write method, optimizer, learning rate, epochs, steps, batch size, seed and trainable count into `result.json` and `model-config.json`.
- **Acceptance check:** two consecutive default runs on the same runtime give identical held-out predictions; the notebook prints trainable/frozen counts that match the prose; `result.json["fine_tuning"]` contains those hyperparameters and the seed.
- **Spec:** FT3, FT5, FT6, ENV7, OUT8.

#### MNV-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; generator `tools/notebook_template.py`.
- **Observed issue:** no intended-audience statement, no "How to use this notebook", no roadmap, no Input → Model → Output contract, no glossary (logits, softmax, argmax, AdamW, cross-entropy, BatchNorm, held-out), no prediction prompt, no interpretation checkpoint, no troubleshooting section, no conclusion template. The 513-line carried module (cell 5) and the manifest cell are not labelled as infrastructure or collapsed. Sections 8 and 9 have no "what to look for" note. The only learner controls are the two BYOD toggles; there is no bounded Predict → Change → Observe → Explain activity.
- **Consequence:** a learner can run the notebook but is not guided to read or question its results, which is exactly where MNV-B1 needed guidance.
- **Evidence:** source inspection (P1 `guided_markers`; the "Predict"/"Checkpoint" hits are the words `predict()` and "checkpoint" as in model weights, not activities; `expected_result_notes_cells_17_19: [false, false]`).
- **Recommended correction:** regenerate with the guided structure the newer fleet notebooks use: audience and how-to-use opening, roadmap, task contract, glossary, a prediction before the fine-tune, an expected-result note after Sections 8 and 9, one safe change-one-thing activity (e.g. epochs or trainable set), troubleshooting, and a conclusion template; label cells 3, 5 and 7 as infrastructure.
- **Acceptance check:** each listed element exists as a markdown cell; cells 3, 5 and 7 carry an infrastructure label; at least one activity asks for a prediction before a run and an explanation after it.
- **Spec:** GDL1–GDL4, GDL6–GDL14, UX4, UX5, UX8.

### Minor

#### MNV-m1 — The split ignores the archive's 100 duplicate pairs and never states the independence assumption

- **Cell/section:** cell 17 split; Section 8 prose.
- **Observed issue:** `Cleanlab/cifar-10-subset` contains `original_images/` and `darkened_images/` copies of the same 100 pictures (pixel-identical after RGB conversion). Both trees are pooled by class name and split at random. The notebook does not mention the darkened variants or say the split assumes independent images.
- **Consequence:** at the default `SUBSET_PER_CLASS = 16` no held-out image has its copy in training (0/6), so today's numbers are not inflated. Any learner who raises the subset (the constant invites it) gets leaked held-out scores: 3/20 at 50 per class, 7/40 at 100, 30/80 at 200.
- **Evidence:** direct (P6: 100 duplicate groups, 200 images; `leak_by_SUBSET_PER_CLASS`).
- **Recommended correction:** group each original with its darkened copy before splitting (or use only `original_images/`), and state the split assumption in Section 8.
- **Acceptance check:** at `SUBSET_PER_CLASS = 200` no held-out image has a pixel-identical copy in training, and the Section 8 prose names the grouping or independence assumption.
- **Spec:** SPL3, SPL10.

#### MNV-m2 — The dataset fallback crashes with `NameError`, and the `except` also swallows a digest mismatch

- **Cell/section:** cell 17 (`except Exception as exc` around download and SHA-256 check; fallback uses `SAMPLE_HEIGHT`/`SAMPLE_WIDTH`). Generator: `tools/notebook_template.py` lines 257 and 322.
- **Observed issue:** the prose promises that an air-gapped runtime "gracefully falls back to deterministic synthetic stripes". With the network unavailable, the cell prints a warning and then raises `NameError: name 'SAMPLE_HEIGHT' is not defined` (neither name is defined anywhere). A tampered download raises the SHA-256 mismatch inside the same `try`, which is caught, printed as a warning, and routed to the same fallback.
- **Consequence:** the promised fallback does not exist, and the learner gets an unrelated `NameError`. If only the missing names were defined, a dataset that fails its digest would be silently replaced with stripes.
- **Evidence:** direct (P7: offline and one-byte-tampered archive both end in `NameError` after a "falling back" warning).
- **Recommended correction:** fail closed on a digest mismatch with an actionable message; either remove the fallback and its promise, or define it properly and label its outputs as a non-default synthetic path.
- **Acceptance check:** a tampered archive raises an error naming the digest mismatch and nothing trains; with no network, the cell either stops with an actionable message or runs a fallback that `result.json` labels as such.
- **Spec:** UX10; data-integrity analogue of MOD8.

#### MNV-m3 — BYOD branches: raw errors, silent rewrites, and no new-data inference

- **Cell/section:** cells 9 and 17 BYOD branches.
- **Observed issue (direct, P8/P9):**
  - Image branch: a non-image file raises a raw `UnidentifiedImageError` without the file name; an empty upload raises a bare `StopIteration`.
  - Dataset branch: an undecodable member raises a raw `UnidentifiedImageError` without the member name; images at the zip root become a class named `unknown` without warning; a class present only in `val/` is dropped without warning; dataset images never pass `validate_inputs` (no manifest, no size ceiling); there is no inference on new data after adaptation. The rules that are enforced (one class, one-image class, `../` member, non-`.zip` name, out-of-range GT, 5000 px image) give clear messages.
- **Consequence:** the BYOD dataset path does not run the full `validate → split → fine-tune → evaluate → new-data inference/export` sequence, and some invalid inputs fail late or are silently reinterpreted.
- **Recommended correction:** validate every dataset image with the pipeline's own `validate_inputs` and write a dataset manifest; name the offending file in decode errors; reject (or explicitly report) root-level images and val-only classes; handle an empty upload; add a new-data inference step after reload.
- **Acceptance check:** each invalid case above stops with a message naming the file or rule; a 2×6 dataset upload produces a dataset input manifest and a prediction on an image outside both splits.
- **Spec:** DAT13, DAT14, DAT19, UX10.

#### MNV-m4 — Prerequisites and release documents contradict what runs

- **Cell/section:** Prerequisites (cell 1); `docs/release-verification.md`; `README.md` line 63; `STATUS.md` line 3; `tutorials/README.md` table.
- **Observed issue:**
  - Prerequisites say "nothing is downloaded" and list the Hub only for the model snapshot; cell 17 downloads the ~986 KB `Cleanlab/cifar-10-subset` archive.
  - `release-verification.md` "Current status" says both "No clean-runtime execution of the notebook has been recorded yet" and that GPU evidence "is now recorded below"; its executor table names a Kaggle CPU kernel while the recorded run is T4; procedure step 5 omits the dataset, fine-tune, reload and held-out evaluation stages; the recorded row says PASSED without the restart (MNV-M1) or the 1/6 result (MNV-B1).
  - `README.md` and `STATUS.md` say the clean-runtime run is pending; `tutorials/README.md` says "verified — clean-runtime `Run all` execution recorded" and lists only single-image BYOD.
- **Consequence:** a reader cannot tell what was verified or what the notebook downloads.
- **Recommended correction:** state the dataset download in Prerequisites; rewrite the release procedure to cover every default stage; record restarts and the fine-tune metrics in the run row; make the four documents agree on one status.
- **Acceptance check:** the four documents give the same status and run description; Prerequisites list every network fetch of the default path.
- **Spec:** REL10, UX12, DAT12.

#### MNV-m5 — "Fresh-boundary reload verification" loads the artifact but never compares it with the trained model

- **Cell/section:** Section 9 (cell 19).
- **Observed issue:** the prose says the reload "verif[ies] artifact integrity"; the cell only evaluates the reloaded model and never compares its outputs with `fine_tuned_pipe`.
- **Consequence:** a broken export that loads cleanly would be reported the same way. In this revision the artifact is faithful (probe max |Δscore| = 0.0), so this is a missing check rather than a wrong result.
- **Evidence:** source; direct (P3).
- **Recommended correction:** compare reloaded and in-memory scores on the held-out images with a stated tolerance and print the maximum difference.
- **Acceptance check:** cell 19 prints a reload-equivalence line with the tolerance and fails when it is exceeded.
- **Spec:** VER4, VER5.

### Suggestions

- **MNV-S1:** regenerate against NOTEBOOK_SPEC 2.2 (the notebook declares 2.0).
- **MNV-S2:** show the held-out images with true label, predicted label and score; with 6 images this makes MNV-B1 visible at a glance.
- **MNV-S3:** report held-out accuracy with a Wilson interval and say that one image is 16.7 points on the default split.
- **MNV-S4:** drop the `mode='RGB'` argument in cell 9's `Image.fromarray` (Pillow deprecation warning in both runs; removal announced for Pillow 13).

## 6. Readiness

**Needs revision.** One Blocker (MNV-B1) and four Majors are open, and applicable MUSTs fail (RUN1, RUN10, ENV6, REL2, DAT5/FT2 in substance, FT3, FT5, FT6, ENV7, OUT8, SPL3, DAT19, VER5, REL10). Remaining gates after fixes: a one-pass hosted `Run all` of the regenerated blob with the fine-tune beating its baselines, recorded with blob id and `restarted: false`; BYOD evidence for REL12.

## 7. Verified versus inferred

- **Verified by direct execution (local CPU, not Colab):** default path 10/10 twice; fine-tune results and spreads; zero-shot grouping 6/6; reload equivalence; duplicate counts; fallback `NameError`; BYOD behaviours listed in MNV-m3.
- **Verified from documented evidence:** the T4 restart and the T4 fine-tune result of 1/6.
- **Inferred:** the cause of the failed adaptation (step count, learning rate, full-network training with batch-4 BatchNorm); that a head-only or BatchNorm-eval fine-tune with more steps would learn this task.
- **Only Kurt can confirm:** Colab behaviour, the upload dialog, and learner understanding.
- **Most likely to be wrong:** the Blocker grading of MNV-B1. The notebook does run end to end and states that sample results are not benchmark evidence; a reviewer who treats the fine-tune as plumbing-only evidence would grade it Major. It is graded Blocker because the notebook's own evaluation shows the adaptation failing and reports `success`, which invalidates the central `E2E` demonstration.
