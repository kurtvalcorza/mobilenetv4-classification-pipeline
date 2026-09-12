# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/tutorials/mobilenetv4_classification_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fmobilenetv4__conv__small.e2400__r224__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)
[![Upstream](https://img.shields.io/badge/Upstream-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white)](https://github.com/huggingface/pytorch-image-models)
[![arXiv](https://img.shields.io/badge/arXiv-2404.10518-b31b1b.svg)](https://arxiv.org/abs/2404.10518)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `mobilenetv4_classification_colab.ipynb` | `TASK-INFERENCE` | ImageNet-1k single-label image classification (1000 classes) with `timm/mobilenetv4_conv_small.e2400_r224_in1k`; argmax decision plus rank-ordered top-5 softmax scores; `top_k_accuracy` only when a ground-truth class index is supplied | CPU (CUDA used automatically when available) CPU is the natural default for this 15 MB model (card-measured CPU smoke 2.08 s load+predict). | single image file, gated off by default; optional ground-truth index | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `MobileNetV4ClassificationPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks, at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (timm executes no remote model code; the notebook never calls `huggingface_hub`).
- The default sample is a synthetic 256×256 RGB gradient generated in code (no download, no ground truth); its prediction is sanity evidence for the input contract and forward pass, not a correctness or benchmark claim. `top_k_accuracy` is reported only when the learner supplies a ground-truth ImageNet-1k class index for a BYOD image.
- Classification obligations (Notebook Specification §20.1): the class count (1000) is printed before the model runs, the decision rule is `argmax`, softmax scores are described as uncalibrated (no threshold is shipped; downstream calibration is the caller's), and exported top-k scores keep their rank ordering in both JSON and CSV.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
