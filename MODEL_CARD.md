---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-classification
base_model: timm/mobilenetv4_conv_small.e2400_r224_in1k
---

# MobileNetV4-Conv-Small e2400_r224_in1k (DIMER package v0.1.0) — Image Classification

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fmobilenetv4__conv__small.e2400__r224__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white)](https://github.com/huggingface/pytorch-image-models)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2404.10518-b31b1b.svg)](https://arxiv.org/abs/2404.10518)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-mobilenetv4--classification--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/tutorials/mobilenetv4_classification_colab.ipynb) [`mobilenetv4_classification_colab.ipynb`](https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/tutorials/mobilenetv4_classification_colab.ipynb)  
  *ImageNet-1k single-label classification with the pinned `timm/mobilenetv4_conv_small.e2400_r224_in1k` weights (15 MB, CPU-natural): one PIL image → 1000 softmax scores → argmax and rank-ordered top-5; `top_k_accuracy` only when a ground-truth index is supplied.*

---

###### Description

`timm/mobilenetv4_conv_small.e2400_r224_in1k` is the smallest all-convolutional member of the MobileNetV4 family (Qin et al., arXiv:2404.10518), trained on ImageNet-1k by Ross Wightman with `timm` scripts "using hyper-parameters inspired by the MobileNet-V4 paper with `timm` enhancements" (upstream README, which also notes these are the only known MNV4 weights because the official TensorFlow weights are unreleased), pinned here to revision `331fb803779522b685cf942e15f914fb6741c1eb`. The network is a stack of the paper's Universal Inverted Bottleneck and fused inverted-bottleneck blocks behind a `conv_stem`, producing feature maps of 32×112×112, 32×56×56, 64×28×28, 96×14×14 and 960×7×7 at 224 px (upstream README feature-map example), then global pooling and a 1000-way `classifier` layer (snapshot `config.json`). Upstream reports 3.8 M parameters and 0.2 GMACs at 224 px — about one twentieth of a ResNet-50 — which is why this repository is the edge/CPU profile of the DIMER timm set. Inference maps a normalised 3×224×224 tensor to 1000 logits in one forward pass; nothing is adapted or fine-tuned here. What this repository adds is packaging: the `MobileNetV4ClassificationPipeline` class in `src/mobilenetv4_classification_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`), input validation, a fixed output contract and a `top_k_accuracy` helper.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is single-label image classification: input one PIL image or a batch of up to `MAX_BATCH = 64` images; output, per image, the `top_k` (default 5) ImageNet-1k classes with their softmax scores plus the argmax label. Envisioned applications are the ones where a 15 MB checkpoint and sub-second CPU latency matter more than the last few points of accuracy: on-device or edge tagging, CPU-only servers, batch triage of large photo archives before a heavier model, and low-cost baselines in the DIMER workbench. In a larger system the pipeline is an inference component or a pre-filter, not a decision engine; the 960-d pooled features are not exposed by this package (the DINOv2 sibling covers feature extraction).

###### Primary Intended Users

The intended users are machine-learning engineers, data scientists and application developers integrating a small classifier into research prototypes, edge or CPU deployments, or the DIMER model workbench. The pipeline assumes its users understand that the label space is fixed to the 1000 ImageNet-1k classes, that a softmax score is not a calibrated probability, that a 3.8 M-parameter model trades accuracy for speed (upstream 73.8 % top-1 versus 80–84 % for the sibling ResNet-50 and ConvNeXt pipelines), and that any deployment on their own data needs a labelled evaluation set. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not object detection, segmentation, multi-label tagging, OCR, or open-vocabulary classification; anything outside the 1000 ImageNet-1k classes cannot be named. Fine-grained distinctions that the larger siblings get right are more often wrong here; do not substitute this model for them where accuracy is the constraint. Feature extraction is not exposed — use `dinov2-feature-extraction-pipeline`.
2. **Input boundary:** only PIL images are accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; batches above 64 are rejected; every image is resized to 256 px and centre-cropped to 224×224 (`crop_pct = 0.875` from the snapshot `config.json`), so fine detail in large images is lost. The upstream 256-px `test_input_size` is not used by this pipeline. Non-RGB modes are converted to RGB; depth, multispectral and video inputs are unsupported.
3. **Decision boundary:** not for autonomous or high-impact decisions — content moderation takedowns, safety interlocks, medical or forensic triage — without a human reviewing the prediction and a locally measured error rate.

#### Factors

###### Groups

The pipeline is not human-centric: it is an object-centric classifier whose label space contains no person-identity, age, gender or skin-type categories. ImageNet-1k nevertheless contains many images of people, and the dataset has documented label problems (ambiguous, offensive and mislabelled categories in the original hierarchy). Neither the upstream `timm` card nor this repository reports any group-level performance breakdown, and the training data is not group-audited; a small-capacity model can also lose minority-appearance classes first when it trades accuracy for size, and nobody has measured whether it does. The fairness audit therefore transfers to the operator: before deployment, measure `top_k_accuracy` on a labelled sample of your own data stratified by the groups that matter to your application, and treat any material gap as a blocker.

###### Instrumentation

ImageNet-1k images were collected from web image searches (Deng et al., 2009) and are consumer camera photographs of varied, undocumented provenance — many makes of camera, lens and post-processing, mostly JPEG-encoded. The pipeline consumes decoded pixel arrays, so the instrument sits behind PIL: resolution, JPEG compression level, colour profile, white balance and sensor noise all reach the model as changed pixel statistics after the 224-px resize. Edge deployments typically have one fixed camera, which makes instrument drift both more likely to matter and easier to test: validate on frames from that camera. The pipeline does not detect drift, blur, over-exposure or a change of capture device; it only rejects non-image types and images outside the 1–4096 px side range.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is optional; `from_pretrained` picks `cuda:0` when available, else CPU, and runs in float32 on both. Measured on this repository's smoke runs (claude-science WSL venv, one synthetic 256×256 image through `MobileNetV4ClassificationPipeline.from_pretrained(device=...).predict`): on **CPU** (Intel Core Ultra 9 275HX, 24 logical processors visible to WSL, float32, no thread pinning) loading the verified snapshot took 1.90 s and one prediction 0.19 s including transform and first call; on the RTX 5070 Ti the same load took 2.01 s and the first prediction 0.84 s, dominated by CUDA warm-up — for this model a GPU is not needed. Data environment: inputs are assumed to be natural photographs whose subject is one of the 1000 classes, framed roughly as in ImageNet; line drawings, medical scans, satellite tiles, heavy occlusion or unusual viewpoints fall outside that assumption and degrade accuracy in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The only measure the code reports is `top_k_accuracy(predictions, targets, k)` in `pipeline.py`: the fraction of images whose target index appears among the first `k` predicted indices, for any `k` up to the requested `top_k`. It captures discrete correctness of the ranking, which suits a 1000-way single-label classifier where the operational question is "is the right class first, or at least in the shortlist". It says nothing about calibration or per-class behaviour, so a reader using top-1 alone cannot tell whether errors are near-misses (fixable by a shortlist) or confident mistakes. Upstream reports 73.756 % top-1 / 91.422 % top-5 at 224 px and 74.616 % / 92.072 % at 256 px on the ImageNet-1k validation set (upstream README comparison table); this pipeline has not reproduced those numbers and reports no accuracy of its own. Latency, the reason to pick this model, is reported only as the single smoke measurements above, not as a benchmark. The public `evaluation_report(result, targets)` helper is the only reporting path: it emits a machine-readable report whose verdict is `sample-sanity` with `top_k_accuracy` at k=1 and k=5 when ground-truth indices are supplied, and `not-measurable` otherwise, stating in that case what labelled data would make the task measurable.

###### Decision thresholds

The default decision rule is `argmax` over the 1000 softmax scores, exposed as `DECISION_RULE = "argmax"` and reported as `predicted_index` / `predicted_label`; this is an implicit threshold of "highest score wins" with no minimum score. No acceptance threshold was set during development and none is shipped: the softmax score is uncalibrated, so any fixed cut-off would be arbitrary. A deployment that needs an abstain option must choose a score cut-off on its own labelled data, trading the cost of a wrong confident label (false positive) against the cost of an unanswered image (false negative) for its application.

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state; the upstream figures cited above are single validation-set evaluations by the upstream author with no reported interval. Inference is deterministic given the same weights, device and library versions: there is no sampling, dropout is disabled by `model.eval()`, and no seed is required. The two smoke runs illustrate device variability: CPU and CUDA produced the same top-5 ordering with top-1 scores of 0.0248 and 0.0246 — a difference in the fourth decimal from kernel choice, enough to reorder near-tied classes on other inputs. The `score` field is a softmax over logits and is not calibrated; a caller who needs probabilities must fit a calibration map on their own labelled data.

#### Ethical considerations and biases

###### Data

Upstream states the checkpoint was trained on ImageNet-1k (upstream README `datasets` and "Model Details"); the disclosure stops there — no per-image licensing, consent status or demographic composition is given, and ImageNet is known to contain photographs of identifiable people scraped from the web, so the presence of personal data is not ruled out. This repository distributes code, tests and documentation; the 15 MB `model.safetensors` snapshot is git-ignored and staged locally under `weights/mobilenetv4-conv-small/` with a manifest, and no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite and the two smoke runs in this repository. The edge profile makes embedded uses — cameras, kiosks, vehicles — foreseeable; any such use that touches safety is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/mobilenetv4_classification_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/mobilenetv4-conv-small/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded (during this pass it caught a wrong revision constant in the generated module before any weight was read); the Hub path is taken only with `allow_download=True` and then through timm's `hf-hub:<id>@<revision>` form; `trust_remote_code` is never enabled (timm executes no remote code). (2) Input integrity — `_validate` rejects non-PIL inputs, empty or over-size batches, and images outside 1–4096 px before the model runs, and the public `validate_inputs(images, top_k, names=...)` stage routes through the same private check so it raises exactly what `predict` raises while returning a machine-readable input manifest of the schema, ceilings, per-input observations and verdict. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, deterministic preprocessing from the snapshot's `pretrained_cfg`, and `model_id`/`model_revision` in every result. (4) Refusals — no feature-map or training API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation (class re-balancing) is applied because the pipeline does not train.

###### Risks and harms

Overconfidence out of distribution: an unrelated image still yields a top-1 label with a score that can look high — the smoke runs labelled a synthetic colour gradient `nipple` (index 680) at score 0.025, an example of how an off-distribution input can draw a label that is embarrassing or harmful if surfaced to people — and the operator bears the harm when that label is acted on. Bias in the label space and training images: ImageNet's classes and their examples skew toward Western, web-scraped imagery, so objects common elsewhere are more often mislabelled; data subjects and third parties bear the harm when such labels feed downstream systems. Capacity: a 3.8 M-parameter model makes more errors than its siblings, so the same misuse does more damage. Automation bias and silent centre-crop loss apply as for the other classifiers. Likelihood under normal photographic use is moderate and rises sharply off-distribution; magnitude ranges from a wrong tag to a wrongly moderated image.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — its label space cannot do these, and adapting it to try would be a misuse; the edge profile does not change that. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating evidence of what an image contains. Any use that violates the Apache-2.0 terms of the upstream weights or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's output is a coarse object label.

## Immutable provenance

- Model: `timm/mobilenetv4_conv_small.e2400_r224_in1k`
- Revision: `331fb803779522b685cf942e15f914fb6741c1eb`
- Snapshot manifest: `weights/mobilenetv4-conv-small/dimer-base-manifest.json`, `totalBytes` 15236763
- `model.safetensors` SHA-256: `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4` (15223016 bytes)
- `config.json` SHA-256: `ecd20bcf1287aaa88129a736d188904d0b134c96e152b55b241be895657fe41a` (681 bytes)
- Weight format: SafeTensors; loader `timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, pretrained_cfg_overlay={"file": ...})`

## Input/output contract

- `MobileNetV4ClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — pass `device="cpu"` explicitly for the edge profile; the default picks CUDA when present.
- `predict(images, top_k=5)` — `images`: one `PIL.Image.Image` or a sequence of 1–64; sides 1–4096 px; any mode (converted to RGB). Returns `{"predictions": [{"predicted_index", "predicted_label", "top_k": [{"label", "index", "score"}, ...]}, ...], "top_k", "decision_rule", "device", "source", "model_id", "model_revision"}`; `score` is the softmax over 1000 classes.
- `top_k_accuracy(predictions, targets, k=1)` — accepts the `predictions` list above or plain index lists.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32 on both CPU and CUDA; preprocessing resize 256 → centre-crop 224, bicubic, ImageNet mean/std from the snapshot `config.json` (`crop_pct` 0.875; the 256-px `test_input_size` is not used).
- Measured, CPU (claude-science WSL venv, Intel Core Ultra 9 275HX, `HF_HUB_OFFLINE=1`, `device="cpu"`): load 1.90 s, predict 0.19 s, total 2.08 s, top-1 on a synthetic 256×256 gradient image `nipple` (index 680) at score 0.0248.
- Measured, CUDA (same venv, RTX 5070 Ti, separate process): load 2.01 s, predict 0.84 s (first-call warm-up), total 2.85 s, same top-5 ordering, top-1 score 0.0246.
- Tests: `pytest -q -o addopts= tests` — 11 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Qin et al. MobileNetV4 — Universal Models for the Mobile Ecosystem. 2024. https://arxiv.org/abs/2404.10518
- Wightman. PyTorch Image Models. https://github.com/huggingface/pytorch-image-models (doi:10.5281/zenodo.4414861)
- Deng et al. ImageNet: A large-scale hierarchical image database. CVPR 2009. https://doi.org/10.1109/CVPR.2009.5206848
- Upstream card: https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
