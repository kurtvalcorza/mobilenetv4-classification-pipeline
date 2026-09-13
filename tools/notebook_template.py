"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "mobilenetv4_classification_pipeline",
    "repo_name": "mobilenetv4-classification-pipeline",
    "stem": "mobilenetv4_classification",
    "notebook_name": "mobilenetv4_classification_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "MobileNetV4ClassificationPipeline",
    "weights_key": "mobilenetv4-conv-small",
    "runtime_imports": ["torch", "timm"],
    "title": "MobileNetV4-Conv-Small — DIMER image classification tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/tutorials/mobilenetv4_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fmobilenetv4__conv__small.e2400__r224__in1k-ffcc4d?style=flat",
            "https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/huggingface/pytorch-image-models",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2404.10518-b31b1b.svg", "https://arxiv.org/abs/2404.10518"),
    ],
    "capability": "ImageNet-1k single-label image classification (1000 classes) using the pinned `timm/mobilenetv4_conv_small.e2400_r224_in1k` weights",
    "intro": (
        "At inference the network maps one normalized 3×224×224 tensor to 1000 logits in a single forward pass; the "
        "pipeline applies a softmax and reports the argmax class plus the top-k classes with their scores. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens "
        "in this notebook — the upstream checkpoint supplies the weights and the preprocessing configuration, and the "
        "carried pipeline module adds snapshot verification, input validation, a fixed output contract and the "
        "`top_k_accuracy`, `validate_inputs` and `evaluation_report` helpers. The default sample is a synthetic image "
        "generated in code; its prediction is demonstration (plumbing) evidence, not a production-quality or benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, generate a synthetic default input and validate it into an input manifest, "
        "run the supported task, read the argmax decision and the uncalibrated top-k softmax scores correctly, exercise "
        "an optional BYOD path, produce an evaluation report that is `sample-sanity` only when a ground-truth class index "
        "exists and `not-measurable` otherwise, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "object detection, segmentation, multi-label tagging, OCR, open-vocabulary classification, feature/embedding "
        "extraction, or any training. The label space is fixed to the 1000 ImageNet-1k classes; an image whose subject "
        "is outside that space still receives a label."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. **CPU is the natural default for this model:** the checkpoint is 15 MB and the repository's model card records a measured CPU smoke of 2.08 s for load plus one prediction (Intel Core Ultra 9 275HX, WSL venv); CUDA is used automatically when present but is not needed. The pinned `torch==2.14.0` install is the largest download of the run.",
        "- **Knowledge:** basic Python and PIL image handling; what a softmax over class logits is.",
        "- **Data:** the default sample is a deterministic 256×256 RGB gradient generated in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, longest side at most 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a deterministic 256×256 RGB gradient built in code (red ramps left to "
                "right, green top to bottom, blue is their mean), so it needs no download and its SHA-256 is printed for the "
                "record. A gradient is not a photograph of any ImageNet class, so it has **no ground truth**: whatever label "
                "the model returns is a sanity check that the input contract, preprocessing and forward pass work, not a "
                "correctness measurement. BYOD is optional and disabled by default; when enabled, upload one image file and, "
                "if you know its ImageNet-1k class index (0–999), set `GROUND_TRUTH_INDEX` so the evaluation step can compute "
                "`top_k_accuracy`. Leave it at `-1` when the label is unknown. Look for a dictionary naming the sample kind, "
                "its size and digest, and whether a ground-truth index was supplied."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "GROUND_TRUTH_INDEX = -1  # @param {{type:\"integer\"}}\n"
                "SAMPLE_SIDE = 256\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic gradient: no randomness, so no seed is needed and the digest is stable.\n"
                "    ramp = np.linspace(0.0, 255.0, SAMPLE_SIDE)\n"
                "    red = np.tile(ramp, (SAMPLE_SIDE, 1))\n"
                "    green = red.T\n"
                "    blue = (red + green) / 2.0\n"
                "    array = np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8)\n"
                "    image = Image.fromarray(array, mode='RGB')\n"
                "    image_name = f'synthetic_gradient_{{SAMPLE_SIDE}}.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "if GROUND_TRUTH_INDEX != -1 and not 0 <= GROUND_TRUTH_INDEX < NUM_CLASSES:\n"
                "    raise ValueError(f'GROUND_TRUTH_INDEX must be -1 (unknown) or an ImageNet-1k class index in 0..{{NUM_CLASSES - 1}}, got {{GROUND_TRUTH_INDEX}}.')\n"
                "ground_truth = None if GROUND_TRUTH_INDEX == -1 else GROUND_TRUTH_INDEX\n"
                "sample_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': sample_sha256, 'ground_truth_index': ground_truth}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `predict` "
                "applies — type, batch size 1..`MAX_BATCH`, image side 1..`MAX_IMAGE_SIDE` px, `top_k` 1..`NUM_CLASSES` — "
                "and returns an **input manifest** naming the schema and ceilings, each input's observed mode and size, "
                "and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a deliberately oversized image and records the pipeline's own error "
                "message as a finding. Inside the pipeline every accepted image is converted to RGB, resized to 256 px and "
                "center-cropped to 224×224 (`crop_pct = 0.875`, bicubic), so content near the border is cropped away; "
                "nothing else is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'NUM_CLASSES': NUM_CLASSES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}}}})\n"
                "input_manifest = validate_inputs(image, top_k=5, names=[image_name])\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Classify\n\n"
                "`predict` returns, per image, `predicted_index`/`predicted_label` and a `top_k` list of `{{label, index, "
                "score}}` entries **ordered by descending score** — rank position is the class ordering, and the exported "
                "files preserve it. The decision rule is `argmax` over the 1000 softmax scores (`decision_rule` in the "
                "result); the pipeline ships no acceptance threshold, and `score` is a softmax over uncalibrated logits, "
                "**not a calibrated probability**. A deployment that needs an abstain option must choose its own score "
                "cut-off on its own labelled data — downstream calibration is the caller's responsibility. Inference is "
                "deterministic given the same weights, device and library versions (no sampling, `model.eval()`); CPU, GPU "
                "and cuDNN kernel choices can reorder near-tied classes. Look for the ranked top-5 list; on the gradient "
                "expect a low top-1 score spread across unrelated classes."
            ),
            "code": (
                "result = pipe.predict(image, top_k=5)\n"
                "prediction = result['predictions'][0]\n"
                "print({{'decision_rule': result['decision_rule'], 'predicted_index': prediction['predicted_index'], 'predicted_label': prediction['predicted_label'], 'device': result['device'], 'source': result['source']}})\n"
                "for rank, item in enumerate(prediction['top_k'], start=1):\n"
                "    print(f\"{{rank:>2}}. index {{item['index']:>4}}  score {{item['score']:.4f}}  {{item['label']}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. When a "
                "ground-truth class index was supplied in Section 4 it carries `top_k_accuracy` (the repository's metric "
                "helper) at k=1 and k=5 with the verdict `sample-sanity` — a single-image tutorial metric with no dispersion "
                "estimate. On the synthetic default sample no metric exists, so the verdict is `not-measurable` and the "
                "report states what would make the task measurable: labelled photographs with ImageNet-1k class indices, "
                "for example a held-out sample of your own data scored against its majority-class baseline, or the "
                "ImageNet-1k validation set (whose upstream 73.756 % top-1 / 91.422 % top-5 at 224 px is quoted from the "
                "upstream card, not measured here). The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "targets = None if ground_truth is None else [ground_truth]\n"
                "report = evaluation_report(result, targets, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No ground-truth class index was supplied, so top_k_accuracy is not computed; the prediction above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full prediction (argmax decision and the rank-ordered top-k scores), "
                "the evaluation report, the sample identity and digest, the notebook's source (repository, revision, "
                "embedded module digest, generator), the model identifier, the immutable model revision, and the runtime "
                "identity (Python, `torch`, `timm`, device). The rank-ordered top-k table is also written as CSV with "
                "explicit `rank`, `index`, `label` and `score` columns so class ordering survives downstream use. No "
                "credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': sample_sha256, 'ground_truth_index': ground_truth}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'timm': timm.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_top_k.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'index', 'label', 'score'])\n"
                "    for rank, item in enumerate(prediction['top_k'], start=1):\n"
                "        writer.writerow([image_name, rank, item['index'], item['label'], f\"{{item['score']:.6f}}\"])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The predicted label is the argmax of a softmax over the fixed 1000-class ImageNet-1k label space; the `score` "
        "values are uncalibrated softmax outputs, not probabilities of correctness, and the pipeline ships no threshold. "
        "On the synthetic gradient the label is meaningless by construction and the evaluation report says "
        "`not-measurable`; a `top_k_accuracy` value shown for a single BYOD image is tutorial evidence for that one "
        "image and must not be generalized to a domain, camera, or class distribution. Images whose subject is outside "
        "ImageNet-1k, line drawings, medical or satellite imagery, and subjects near the image border (removed by the "
        "center crop) all degrade results in ways the pipeline does not detect. The pipeline provides no detection, "
        "segmentation, multi-label, OCR, feature extraction, or training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, and emit the shown machine-readable outputs in the tested runtime — without the repository being "
        "reachable. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a photograph of a known ImageNet class and its index to see the "
        "report switch to `sample-sanity` with `top_k_accuracy` at k=1 and k=5; classify a batch (up to `MAX_BATCH`) of "
        "labelled images from your own domain and compare top-1 against the majority-class baseline of that set; inspect "
        "how the top-1 score moves when the subject is moved toward the border of the frame.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/huggingface/pytorch-image-models\n"
        "- MobileNetV4 — Universal Models for the Mobile Ecosystem: https://arxiv.org/abs/2404.10518"
    ),
}
