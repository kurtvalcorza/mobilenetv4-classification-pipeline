# mobilenetv4_classification_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`fcb4766`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers and bootstrap check); `docs/release-verification.md` (one line describing the check) | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED mode with 1 of 9 guided markers. Added a model-specific guided layer: audience and Input → Model → Output table, How to use this notebook, roadmap, a Learner prerequisite, five **Predict before running** prompts with **Check your reasoning** answers (Sections 4, 6, 7, 8, 9), Troubleshooting, Glossary and a Conclusion template. Sections 1–3 are labelled Infrastructure and their cells collapsed. No hosted per-cell outputs are recorded in this repository, so the worked answers state what the code guarantees (for example the default split of 26 training / 6 validation images and the exact 50 % majority baseline, both computed from the code) and quote no measured accuracy. | Template opening, prerequisites, Sections 4 and 6–9, closing | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders` |
| SWP-A (quality asserts) | Not applicable | The sweep found 0 quality asserts; none present. | — | sweep row `quality_asserts = 0` |
| SWP-F (frozen re-run) | Not applicable | `fit` is a classmethod that builds a new model from the verified snapshot; the pretrained `pipe` is never trained in place. | — | `src/.../pipeline.py` `fit` |
| SWP-B (BYOD) | Fixed | Both BYOD branches worked only through `google.colab.files.upload()`. Added `BYOD_PATH` (Section 4) and `BYOD_DATASET_PATH` (Section 8) form fields that work on Colab, Kaggle and Jupyter, with the Colab upload as a guarded fallback (an empty or cancelled upload, or an empty path off Colab, gives a clear message). Refusals name the file or archive member and the rule; an undecodable image no longer raises a bare Pillow error; unused archive members and validation classes absent from `train/` are counted and printed (`skipped_members`) instead of silently dropped; an explicit `val/` with no usable image is refused; the effective minimum (2 classes, 2 images per class when un-split) is stated in the Prerequisites. Also fixed in the same cell: the offline synthetic-stripe fallback referenced undefined `SAMPLE_HEIGHT`/`SAMPLE_WIDTH` (a `NameError` whenever the sample download failed); it now uses `SAMPLE_SIDE`. The default-path split order is unchanged. | Sections 4 and 8, prerequisites, BYOD declaration | `test_swp_b_byod_path_fields_default_off`, `test_swp_b_byod_path_reads_file_and_refuses_with_names`, `test_swp_b_cancelled_colab_upload_gives_a_clear_message`, `test_swp_b_dataset_split_counts_and_reports_skipped_members`, `test_swp_b_dataset_refusals_name_the_rule` (5 cases), `test_swp_b_synthetic_fallback_has_no_undefined_names` |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart. It builds (first run) or reuses
  a separate environment `dimer_isolated_env_<lock digest>/` and every later code cell runs there. Linux x86_64 runtimes
  only (Colab, Kaggle, Linux Jupyter); other platforms stop with that message.
- New form fields `BYOD_PATH` (Section 4) and `BYOD_DATASET_PATH` (Section 8). On Colab an empty path still opens the
  upload dialog.
- Section 8 prints `dataset_source`, the image counts and `skipped_members` before training.
- New guided material (predictions, worked answers, Troubleshooting, Glossary, Conclusion); Sections 1–3 collapsed.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages could run here (the Hugging Face Hub is unreachable and torch is not installed).
- Synthetic data: the BYOD tests build PNGs and zip archives in memory and execute the notebook's own Section 4 helper
  and Section 8 dataset code (up to the `fit` call).
- Stand-in: `test_swp_r_section1_reuses_environment_and_worker_when_rerun` executes the generated kernel cell against a
  pre-built environment folder whose `python` is the test interpreter, so the routing, reuse and idempotence logic is
  real while the managed CPython and the locked packages are stand-ins.
- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's lightweight dependencies (torch absent, so `tests/test_pipeline.py`, which imports torch at module
  level, was excluded both before and after): 16 passed before → 33 passed after. CI itself installs CPU torch and
  also runs `tests/test_pipeline.py`, which this change does not touch.
- Every code cell of the regenerated notebook parses (`ast.parse`, 11 code cells).

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment;
  a second Run all reports `'reused': True`).
- The REL12 BYOD run with `USE_BYOD`/`BYOD_PATH` and `USE_BYOD_DATASET`/`BYOD_DATASET_PATH`.
- A full Notebook Review Framework v1 review has not been done.
