# Release verification

`tutorials/mobilenetv4_classification_colab.ipynb` (`E2E`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, `standalone: true` and `generated_from` (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/mobilenetv4_classification_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the single kernel cell that builds (or reuses, by lock digest) the isolated hash-locked uv environment and routes
  every later cell to it, with no `pip install` into the kernel and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `MobileNetV4ClassificationPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `predict`, `pipe.fit`, fresh-boundary reload verification, `evaluation_report`), the ceiling print (`NUM_CLASSES`, `MAX_IMAGE_SIDE`, `MAX_BATCH`),
  the exports, the learner-facing classification statements (argmax decision rule, uncalibrated
  softmax, no shipped threshold, rank-ordered scores) and the gated-off BYOD default listed in
  the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `timm.create_model` / `from timm import` / `from torchvision import` /
  `from transformers import` / `from huggingface_hub import` use **outside the carried module cell**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchvision` wheels plus `timm`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `GROUND_TRUTH_INDEX = -1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - synthetic 256×256 gradient sample generated in code with its RGB SHA-256 printed and the
     ceilings (`NUM_CLASSES` 1000, `MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 64) surfaced;
   - pinned `timm/mobilenetv4_conv_small.e2400_r224_in1k` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/mobilenetv4-conv-small/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all three manifest entries
     (`README.md`, `config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports
     `source == 'local-snapshot'`;
   - classification through `predict(image, top_k=5)` with `decision_rule == 'argmax'` and a
     rank-ordered top-5 list;
   - `validate_inputs` writes `outputs/mobilenetv4_classification_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the oversized probe);
   - `evaluation_report` writes `outputs/mobilenetv4_classification_evaluation_report.json` with verdict `not-measurable`
     on the synthetic sample (no ground truth), stated as such;
   - `outputs/mobilenetv4_classification_result.json` and `outputs/mobilenetv4_classification_top_k.csv`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/mobilenetv4_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/mobilenetv4_classification_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `258fecf` / `484cc1023f6e` | Kaggle T4 (`kurtvalcorza/dimer-nb2-mobilenetv4-classification` v1) | Default sample path | 168.9 s | **PASSED** — 10/10 ok code cells executed cleanly, 8 files, 15 MB staged |

### Hosted Colab CLI runs (isolated environment)

Executor: Colab CLI 0.7.4 sequential execution (`colab exec -f`) on a fresh Colab Tesla T4 VM, driven by the
workspace serial suite, which fetches the notebook byte-exact at the commit and refuses it unless its Git blob matches.
This is not a browser Run all: the executed file carries no execution counts, and cell order is evidenced by
`exec.log` ("Executing cell k/N"). Up to the 2026-10-07 run only the default path ran, and the 2026-10-30 refusal run adds only a scripted DAT20 refusal (an unsupported file, refused before any model call); the upload/BYOD branches (`USE_BYOD`,
`USE_BYOD_DATASET`) and the optional "Change one thing" activity were not exercised.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-30 | `abe8dfbff7eca13233dd3de51d274abb172f411b` (PR #11 head) / `mobilenetv4_classification_colab.ipynb` blob `621cabc6d2bbec5e49bf60569b43684ec34b43aa` (current blob) | Google Colab CLI sequential execution (`colab exec -f`), fresh Colab Tesla T4 (session `suite-mobilenetv4-abe8dfb-b032`): every code cell run in order in one kernel; not a browser `Run all`, and the CLI records no execution counts, so order is evidenced by `exec.log` (cells 1–11 in order); kernel Python 3.13.16, isolated CPython 3.12.12 (45 locked packages, 57 s); the committed blob fetched at the 40-character SHA and Git-blob verified, input unchanged | Default path only, form fields at their defaults, no overrides | 111.2 s | **One pass, no restart, 0 errors**; 11/11 code cells; VM released (`stop_verified`); the executed notebook's SHA-256 equals `run_summary.json` `executed_sha256`. Synthetic gradient image predicted index 818 `spotlight, spot` (score 0.0322) on `cuda:0`, single-image report verdict `not-measurable`; Cleanlab/cifar-10-subset frog/truck 160 train / 40 held-out images, head fine-tuned 8 epochs (2,562 trainable parameters), `loss_fell_below_chance` `True`; held-out accuracy 0.975, `Majority Baseline:  50.00% (predicting 'truck')`, `Zero-shot reference: 100.00%`, `Lift Over Baseline: +47.50%`, `VERDICT: above-majority-baseline | fine-tuned model BELOW the zero-shot reference (adaptation lost ground)`; `reload equivalence: max |score difference| over 40 held-out images = 0.00e+00 (tolerance 1e-05)`; `model.safetensors` and `model-config.json` written under `outputs/mobilenetv4_classification_finetuned/`. Evidence `docs/execution-evidence/2026-10-30-abe8dfb/`: SHA-256 captured → committed: `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_default.ipynb` `acf66805ccb8924a4e303d10d1fad33e5317d3fcdf4878ad9ea9d2bcabca83e7` → unchanged (no local path); `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_default_exec.log` `21b18f3a2f20159730f54e9cc73b73f7595ad0dd86085319462979c0852d068a` → `cc7895df8c6053f2bd92067587818343f4137e0aebce4347e0b2636ac361d43e` (2 local-path replacements); `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_default_run_summary.json` `b49b824899e52d809960ffeb46f95c29f2d59aadbd4231ddd94efe6d6e9ec59f` → unchanged (no local path). Not exercised: an accepted BYOD input, optional switches off their defaults, and the form widgets |
| 2026-10-30 | same | same, fresh VM (session `suite-mobilenetv4-abe8dfb-b032`) | Scripted DAT20 refusal: `USE_BYOD = True`; `BYOD_PATH = '/etc/hostname'` (set by `--set` on a copy; the run copy differs from the blob in exactly those 2 source lines), and the run copy cut right after the refusing cell `mobilenetv4_classification-12`, dropping 18 cells (6 code cells), which reproduces `Run all` stopping at the first error | 107.6 s | **Refused as designed** — 5/5 code cells of the cut copy. The 4 earlier code cells completed without an error; the refusing cell printed exactly one `Refused:` line, with no traceback: `Refused: hostname: Pillow cannot decode this file (cannot identify image file <_io.BytesIO object at 0x7a90d05c3600>); BYOD needs one PNG, JPEG, WebP or similar image; fix the input and run this cell again (the earlier results of this session and their outputs/ files were removed).`. The suite recorded the halting error as `IsolatedCellRefusal`. `verdict` `PASS`, input unchanged, VM released. Cut copy `run_copy.ipynb` SHA-256 `7201dfc5ec63c5e6a5de4e4a447baf9e59cb630341e888242de34d015345417d` (Git blob `061b5ca6`), equal to the summary's `run_copy_sha256`. This run says nothing about the 18 cut cells, which did not run in it. Evidence `docs/execution-evidence/2026-10-30-abe8dfb/`: SHA-256 captured → committed: `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_refusal.ipynb` `53981224d472660f3861335e4207c7c2c19400b85bde8c7bd6ede00e852423cb` → unchanged (no local path); `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_refusal_exec.log` `1eec8dfcef529b62d01329050de8f5479f1dcadd1c23e7dc42366535cf039085` → `6ce196548a8df3834bcb9bb4214e62bda904af2301be1f799ff1f71e69b0df16` (2 local-path replacements); `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_refusal_run_summary.json` `35cf07e05ce43cc5f9e4602d68b7b518accd736501e0527dd3b1280ac0149504` → unchanged (no local path); `mobilenetv4_classification_colab_abe8dfb_colab-cli-t4_refusal_run_copy.ipynb` `7201dfc5ec63c5e6a5de4e4a447baf9e59cb630341e888242de34d015345417d` → unchanged (no local path) |
| 2026-10-07 | `a8a888d` / `b622b4fb0c2c` (generator /2.2: fleet-sweep and MNV review fixes, worker `google.colab` stubs with module specs) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-mobilenetv4-a8a888d-b032`) | Default path, `USE_BYOD = False`, `USE_BYOD_DATASET = False`, `TRAINABLE = "head"`: isolated env (45 locked packages, Python 3.12.12, built in 60 s, `reused: False`; kernel Python 3.13.15); worker routed (`worker_reused: False`); snapshot staged and digest-verified, `source == 'local-snapshot'` on `cuda:0`; synthetic gradient → `spotlight, spot` (index 818, score 0.0322), input manifest `accepted` with the oversized probe `rejected`, evaluation report `not-measurable`; `Cleanlab/cifar-10-subset` digest matched, 160 train / 40 held-out images (`frog`, `truck`); head-only fit, 2,562 trainable / 2,493,024 frozen parameters, 160 AdamW steps at 1e-3 in 7.2 s, training loss 0.384 → 0.084 (below ln 2); reload equivalence max score difference 0.0; held-out 39/40 = 97.5 % (95 % Wilson 87.1–99.6 %), majority baseline 50.0 %, zero-shot reference 100.0 %, verdict `above-majority-baseline`, below the zero-shot reference; 7 outputs written. Code cell 3 (carried module) prints nothing by design | 100.5 s | **PASSED** — 11/11 code cells in one pass, no restart, 0 error outputs; code-cell sources identical to the blob. Evidence: `docs/execution-evidence/2026-10-07-a8a888d/` — `mobilenetv4_classification_colab_a8a888d_colab-cli-t4.ipynb` (SHA-256 `087cea03e30fd5c8ed826e5704ef8e253e591a1c677aa442724e215d73f10037`), `run_summary.json` (`8fb38779ef3adf8d5d086f3070f42a9376f4aa04de5c766f74bfea322766a786`), `exec.log` (`ac19b7c451a52c07dd4753dc0127f3b3fd283772528ad231c7e9593353ecf3d3`) |

## Current status

The current blob `621cabc6` (code commit `40e0db0`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); on 2026-10-30 its default path (11/11 code cells, one pass, no restart, 0 errors) and its scripted DAT20 refusal (`USE_BYOD=True;BYOD_PATH='/etc/hostname'`: exactly one `Refused:` line at cell `mobilenetv4_classification-12`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI sequential execution, not a browser `Run all`; see "Recorded executions" in `docs/release-verification.md`), and the hosted runs of earlier blobs, including `b622b4fb`, are history for it (REL14). Before that change: the earlier blob `b622b4fb0c2c` (commit `a8a888d`, history for the current blob, REL14) passed a Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (2026-10-07, 11/11 code cells, one pass, no restart, 0 errors, 100.5 s; held-out 97.5 % against a 50 % majority baseline and a 100 % zero-shot reference; see "Hosted Colab CLI runs"). It exercised the default path only, not a browser Run all; the BYOD branches and the optional activity have not been executed in any record. The earlier Kaggle T4 run of blob `484cc1023f6e` predates the isolated environment and the review fixes. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Two facts a reviewer should
weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite (the
real `hf_hub_download` fetch of all three manifest entries into a fresh `weights/mobilenetv4-conv-small/` has not been
executed), and the standalone carrier itself — executing the carried module cell in a runtime that has no
repository checkout — has been validated statically only (parity PASS), never run; the earlier local smoke run
used the verified snapshot through the installed package, so the clean run will be the first execution
of the standalone path, of the staging path, and of the CPU inference path against the real weights.