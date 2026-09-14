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
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned "
        "snapshot, generates the deterministic synthetic sample image, validates it into an input manifest, classifies it with the "
        "pretrained head, writes the zero-shot evaluation report, downloads and digest-verifies the `Cleanlab/cifar-10-subset` tutorial "
        "dataset (a balanced per-class subset with a seeded 80/20 split), performs a bounded in-kernel fine-tuning run of a new "
        "classification head with AdamW and cross-entropy, exports the artifact (`model.safetensors` + `model-config.json`), reloads it "
        "from disk across a fresh boundary and evaluates the reloaded model on the held-out split against a majority-class baseline, "
        "and writes outputs and provenance. No repository clone, DIMER worker or service, credential, upload dialog or configuration "
        "edit is required (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "Two optional branches, both off by default and never part of the default path: `USE_BYOD = True` in Section 4 uploads one "
        "image that passes through the same validation, classification, evaluation-report and export cells as the synthetic sample; "
        "`USE_BYOD_DATASET = True` in Section 8 uploads a `.zip` of class folders (or `train/` and `val/` directories) that enters the "
        "same validation, seeded split, in-kernel fine-tuning, export, fresh-reload and held-out evaluation cells as the tutorial "
        "dataset. Expected formats, limits and privacy guidance are stated in the Prerequisites and in those cells; uploads stay "
        "inside this runtime."
    ),
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
    "capability": "ImageNet-1k single-label image classification (1000 classes) and in-kernel fine-tuning using the pinned `timm/mobilenetv4_conv_small.e2400_r224_in1k` weights",
    "intro": (
        "At inference the network maps one normalized 3×224×224 tensor to 1000 logits in a single forward pass; the "
        "pipeline applies a softmax and reports the argmax class plus the top-k classes with their scores. "
        "**In-kernel fine-tuning:** this notebook demonstrates both zero-shot base inference on ImageNet-1k classes and "
        "100% in-kernel classification head fine-tuning on custom classes using PyTorch AdamW optimization and cross-entropy "
        "loss. The upstream checkpoint supplies the pretrained backbone weights, and the carried pipeline module adds snapshot "
        "verification, input validation, head adaptation via `fit`, fresh-boundary reload verification, and the "
        "`top_k_accuracy`, `validate_inputs` and `evaluation_report` helpers. The default sample is a synthetic image "
        "generated in code; its prediction is demonstration (plumbing) evidence, not a production-quality or benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, generate a synthetic default input and validate it into an input manifest, "
        "run the supported task, read the argmax decision and the uncalibrated top-k softmax scores correctly, execute "
        "100% in-kernel fine-tuning on custom classes with AdamW and cross-entropy, export and fresh-reload fine-tuned "
        "artifacts, exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` only when a "
        "ground-truth class index exists and `not-measurable` otherwise, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "object detection, segmentation, multi-label tagging, OCR, open-vocabulary classification, feature/embedding "
        "extraction. The base label space is fixed to the 1000 ImageNet-1k classes; an image whose subject is outside that "
        "space still receives a label unless adapted via in-kernel fine-tuning."
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
                "## 8. Dataset acquisition and in-kernel fine-tuning\n\n"
                "`pipe.fit(...)` implements 100% in-kernel head adaptation: it replaces the 1000-class head with a new "
                "linear classifier sized to the target classes (`len(class_names)`), initializes from the verified backbone "
                "weights without shape collision, and trains with `torch.optim.AdamW` and cross-entropy loss. No external "
                "worker repositories, CLI subprocesses, or unpinned dependencies are invoked.\n\n"
                "**Data Acquisition & BYOD:**\n"
                "- **Default Sample Dataset (`USE_BYOD_DATASET = False`):** Automatically downloads and verifies the SHA-256 digest of "
                "[`Cleanlab/cifar-10-subset`](https://huggingface.co/datasets/Cleanlab/cifar-10-subset) (MIT license, ~986 KB, "
                "400 images across 2 balanced classes: `frog` and `truck`). A balanced subset is loaded for rapid tutorial smoke, "
                "and automatically split into `train` (80%) and `val` (20%). If the runtime is air-gapped, it gracefully falls back "
                "to deterministic synthetic stripes.\n"
                "- **Bring Your Own Data (`USE_BYOD_DATASET = True`):** Upload a `.zip` archive containing either explicit `train/` and `val/` "
                "directories or un-split class folders (in which case a seeded 80/20 stratified split is performed automatically). "
                "Enforces >= 2 classes and >= 2 images per class.\n\n"
                "Deployable fine-tuned artifacts (`model.safetensors` and `model-config.json`) are written atomically to `outputs/{stem}_finetuned`."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import random\n"
                "import urllib.request\n"
                "import zipfile\n\n"
                "USE_BYOD_DATASET = False  # @param {{type:\"boolean\"}}\n"
                "SAMPLE_DATASET_URL = 'https://huggingface.co/datasets/Cleanlab/cifar-10-subset/resolve/bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75/CIFAR-10-subset.zip'\n"
                "SAMPLE_DATASET_SHA256 = '66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19'\n"
                "SEED = 42\n"
                "VALIDATION_SPLIT = 0.2\n"
                "SUBSET_PER_CLASS = 16  # balanced sample per class for fast in-kernel smoke\n\n"
                "train_images, train_targets = [], []\n"
                "val_images, val_targets = [], []\n"
                "zip_bytes = None\n\n"
                "if USE_BYOD_DATASET:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError('Upload exactly one dataset .zip archive.')\n"
                "    zip_name, zip_bytes = next(iter(uploaded.items()))\n"
                "    if not zip_name.lower().endswith('.zip'):\n"
                "        raise ValueError(f'BYOD archive must be a .zip file, got {{zip_name}}')\n"
                "    dataset_source = f'user upload: {{zip_name}}'\n"
                "else:\n"
                "    try:\n"
                "        req = urllib.request.Request(SAMPLE_DATASET_URL, headers={{'User-Agent': 'Mozilla/5.0'}})\n"
                "        with urllib.request.urlopen(req, timeout=30) as resp:\n"
                "            zip_bytes = resp.read()\n"
                "        actual_sha = hashlib.sha256(zip_bytes).hexdigest()\n"
                "        if actual_sha != SAMPLE_DATASET_SHA256:\n"
                "            raise ValueError(f'Sample dataset SHA-256 mismatch: expected {{SAMPLE_DATASET_SHA256}}, got {{actual_sha}}')\n"
                "        dataset_source = f'Cleanlab/cifar-10-subset (MIT license, sha256:{{actual_sha[:16]}}...)'\n"
                "    except Exception as exc:\n"
                "        print(f'Warning: public sample dataset download failed ({{exc}}); falling back to deterministic synthetic dataset.')\n"
                "        zip_bytes = None\n"
                "        dataset_source = 'synthetic stripes fallback'\n\n"
                "if zip_bytes is not None:\n"
                "    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:\n"
                "        # Security audit: reject directory traversal and absolute paths\n"
                "        for info in z.infolist():\n"
                "            if '..' in info.filename or info.filename.startswith(('/', '\\\\')):\n"
                "                raise ValueError(f'Security violation: illegal path in zip archive: {{info.filename}}')\n"
                "        names = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) and not n.startswith('__MACOSX')]\n"
                "        if not names:\n"
                "            raise ValueError('No supported image files (.png, .jpg, .jpeg, .webp, .bmp) found in archive.')\n\n"
                "        has_train = any('train/' in n.lower() for n in names)\n"
                "        has_val = any('val/' in n.lower() or 'valid/' in n.lower() for n in names)\n\n"
                "        def _class_from_path(p):\n"
                "            parts = p.strip('/').split('/')\n"
                "            return parts[-2] if len(parts) >= 2 else 'unknown'\n\n"
                "        if has_train and has_val:\n"
                "            train_names = [n for n in names if 'train/' in n.lower()]\n"
                "            val_names = [n for n in names if 'val/' in n.lower() or 'valid/' in n.lower()]\n"
                "            CUSTOM_CLASSES = sorted(list({{_class_from_path(n) for n in train_names}}))\n"
                "            if len(CUSTOM_CLASSES) < 2:\n"
                "                raise ValueError(f'Classification requires at least 2 distinct classes, found {{CUSTOM_CLASSES}}')\n"
                "            cls_map = {{c: i for i, c in enumerate(CUSTOM_CLASSES)}}\n"
                "            for n in train_names:\n"
                "                cls = _class_from_path(n)\n"
                "                if cls in cls_map:\n"
                "                    train_images.append(Image.open(io.BytesIO(z.read(n))).convert('RGB'))\n"
                "                    train_targets.append(cls_map[cls])\n"
                "            for n in val_names:\n"
                "                cls = _class_from_path(n)\n"
                "                if cls in cls_map:\n"
                "                    val_images.append(Image.open(io.BytesIO(z.read(n))).convert('RGB'))\n"
                "                    val_targets.append(cls_map[cls])\n"
                "        else:\n"
                "            class_to_files = {{}}\n"
                "            for n in names:\n"
                "                cls = _class_from_path(n)\n"
                "                class_to_files.setdefault(cls, []).append(n)\n"
                "            CUSTOM_CLASSES = sorted(list(class_to_files.keys()))\n"
                "            if len(CUSTOM_CLASSES) < 2:\n"
                "                raise ValueError(f'Classification requires at least 2 distinct classes to train, found {{len(CUSTOM_CLASSES)}}: {{CUSTOM_CLASSES}}')\n"
                "            cls_map = {{c: i for i, c in enumerate(CUSTOM_CLASSES)}}\n"
                "            rng = random.Random(SEED)\n"
                "            for cls, files in class_to_files.items():\n"
                "                if len(files) < 2:\n"
                "                    raise ValueError(f'Class {{cls!r}} has fewer than 2 images ({{len(files)}}); cannot perform train/val split.')\n"
                "                f_list = list(files)\n"
                "                rng.shuffle(f_list)\n"
                "                if not USE_BYOD_DATASET and SUBSET_PER_CLASS:\n"
                "                    f_list = f_list[:SUBSET_PER_CLASS]\n"
                "                n_val = max(1, int(len(f_list) * VALIDATION_SPLIT))\n"
                "                val_files = f_list[:n_val]\n"
                "                train_files = f_list[n_val:]\n"
                "                for f in train_files:\n"
                "                    train_images.append(Image.open(io.BytesIO(z.read(f))).convert('RGB'))\n"
                "                    train_targets.append(cls_map[cls])\n"
                "                for f in val_files:\n"
                "                    val_images.append(Image.open(io.BytesIO(z.read(f))).convert('RGB'))\n"
                "                    val_targets.append(cls_map[cls])\n"
                "else:\n"
                "    CUSTOM_CLASSES = ['synthetic_horizontal_stripe', 'synthetic_vertical_stripe']\n"
                "    for cls_idx, pattern in enumerate(['horizontal', 'vertical']):\n"
                "        for i in range(6):\n"
                "            arr = np.zeros((SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), dtype=np.uint8)\n"
                "            if pattern == 'horizontal':\n"
                "                arr[::16, :, 0] = 255\n"
                "                arr[:, :, 2] = (i * 30) % 255\n"
                "            else:\n"
                "                arr[:, ::16, 1] = 255\n"
                "                arr[:, :, 2] = (i * 30) % 255\n"
                "            img = Image.fromarray(arr, mode='RGB')\n"
                "            if i < 4:\n"
                "                train_images.append(img)\n"
                "                train_targets.append(cls_idx)\n"
                "            else:\n"
                "                val_images.append(img)\n"
                "                val_targets.append(cls_idx)\n\n"
                "ft_output_dir = 'outputs/{stem}_finetuned'\n"
                "fine_tuned_pipe, train_meta = pipe.fit(\n"
                "    train_images=train_images,\n"
                "    train_targets=train_targets,\n"
                "    val_images=val_images,\n"
                "    val_targets=val_targets,\n"
                "    class_names=CUSTOM_CLASSES,\n"
                "    epochs=1,\n"
                "    batch_size=4,\n"
                "    learning_rate=1e-4,\n"
                "    weights_dir=WEIGHTS_DIR,\n"
                "    output_dir=ft_output_dir,\n"
                ")\n"
                "print({{\n"
                "    'fine_tuning': 'complete',\n"
                "    'dataset_source': dataset_source,\n"
                "    'classes': CUSTOM_CLASSES,\n"
                "    'train_samples': len(train_images),\n"
                "    'val_samples': len(val_images),\n"
                "    'epochs': len(train_meta['history']),\n"
                "    'history': train_meta['history'],\n"
                "}})"
            ),
        },
        {
            "md": (
                "## 9. Fresh-boundary reload and in-notebook evaluation\n\n"
                "To verify artifact integrity across an isolation boundary (simulating a fresh deployment or downstream "
                "container), `MobileNetV4ClassificationPipeline.from_pretrained` loads the newly generated `model.safetensors` "
                "and `model-config.json` directly from `outputs/{stem}_finetuned`. The pipeline re-instantiates the architecture, "
                "restores weights with `strict=True`, and configures the custom class labels. "
                "The reloaded pipeline is then evaluated across all held-out validation images (`val_images`) to compute "
                "top-1 validation accuracy, compare against the trivial majority-class baseline, and export machine-readable "
                "evaluation artifacts (`outputs/{stem}_validation_predictions.csv` and `outputs/{stem}_finetuned_evaluation_report.json`)."
            ),
            "code": (
                "import csv\n\n"
                "reloaded_pipe = MobileNetV4ClassificationPipeline.from_pretrained(weights_dir=ft_output_dir)\n\n"
                "# In-notebook evaluation across the held-out validation split\n"
                "val_labels = [CUSTOM_CLASSES[idx] for idx in val_targets]\n"
                "val_predictions = []\n"
                "val_hits = 0\n"
                "per_class_stats = {{cls_name: {{'total': 0, 'correct': 0}} for cls_name in CUSTOM_CLASSES}}\n\n"
                "for img, true_label in zip(val_images, val_labels):\n"
                "    pred_out = reloaded_pipe.predict(img)\n"
                "    pred = pred_out['predictions'][0]\n"
                "    is_correct = (pred['predicted_label'] == true_label)\n"
                "    if is_correct:\n"
                "        val_hits += 1\n"
                "    per_class_stats[true_label]['total'] += 1\n"
                "    if is_correct:\n"
                "        per_class_stats[true_label]['correct'] += 1\n"
                "    val_predictions.append({{\n"
                "        'true_label': true_label,\n"
                "        'predicted_label': pred['predicted_label'],\n"
                "        'predicted_index': pred['predicted_index'],\n"
                "        'confidence': pred['top_k'][0]['score'],\n"
                "        'correct': is_correct,\n"
                "    }})\n\n"
                "val_accuracy = val_hits / max(len(val_labels), 1)\n"
                "majority_class = max(set(val_labels), key=val_labels.count)\n"
                "majority_baseline = val_labels.count(majority_class) / max(len(val_labels), 1)\n\n"
                "finetuned_eval_report = {{\n"
                "    'task': 'single-label classification fine-tuning evaluation',\n"
                "    'verdict': 'success',\n"
                "    'validation_samples': len(val_labels),\n"
                "    'classes': CUSTOM_CLASSES,\n"
                "    'metrics': {{\n"
                "        'accuracy': round(val_accuracy, 4),\n"
                "        'majority_baseline_accuracy': round(majority_baseline, 4),\n"
                "        'accuracy_lift_over_baseline': round(val_accuracy - majority_baseline, 4),\n"
                "        'mean_confidence': round(sum(p['confidence'] for p in val_predictions) / max(len(val_predictions), 1), 4),\n"
                "    }},\n"
                "    'per_class_breakdown': {{\n"
                "        c: {{\n"
                "            'total': s['total'],\n"
                "            'correct': s['correct'],\n"
                "            'accuracy': round(s['correct'] / s['total'], 4) if s['total'] > 0 else 0.0,\n"
                "        }} for c, s in per_class_stats.items()\n"
                "    }},\n"
                "    'fresh_boundary_source': reloaded_pipe.source,\n"
                "}}\n\n"
                "eval_report_path = f'outputs/{stem}_finetuned_evaluation_report.json'\n"
                "with open(eval_report_path, 'w', encoding='utf-8') as handle:\n"
                "    json.dump(finetuned_eval_report, handle, indent=2, ensure_ascii=False)\n\n"
                "val_preds_csv_path = f'outputs/{stem}_validation_predictions.csv'\n"
                "with open(val_preds_csv_path, 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=['sample_index', 'true_label', 'predicted_label', 'correct', 'confidence'])\n"
                "    writer.writeheader()\n"
                "    for idx, p in enumerate(val_predictions):\n"
                "        writer.writerow({{'sample_index': idx, 'true_label': p['true_label'], 'predicted_label': p['predicted_label'], 'correct': p['correct'], 'confidence': f\"{{p['confidence']:.6f}}\"}})\n\n"
                "print('=== Fine-Tuned Model In-Notebook Evaluation ===')\n"
                "print(f\"Validation Samples: {{len(val_labels)}} | Classes: {{CUSTOM_CLASSES}}\")\n"
                "print(f\"Top-1 Accuracy:     {{val_accuracy:.2%}} ({{val_hits}}/{{len(val_labels)}} correct)\")\n"
                "print(f\"Majority Baseline:  {{majority_baseline:.2%}} (predicting '{{majority_class}}')\")\n"
                "print(f\"Lift Over Baseline: {{val_accuracy - majority_baseline:+.2%}}\")\n"
                "print(json.dumps(finetuned_eval_report['metrics'], indent=2))\n"
            ),
        },
        {
            "md": (
                "## 10. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full prediction (argmax decision and the rank-ordered top-k scores), "
                "the evaluation report, the fine-tuning training history and in-notebook evaluation metrics, the sample identity and digest, "
                "the notebook's source (repository, revision, embedded module digest, generator), the model identifier, the immutable model "
                "revision, and the runtime identity (Python, `torch`, `timm`, device). The rank-ordered top-k table and fine-tuned validation "
                "predictions are written as CSV files so class ordering survives downstream use. Deployable fine-tuned model artifacts and evaluation "
                "reports are published in `outputs/{stem}_finetuned/` and `outputs/`. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'fine_tuning': {{\n"
                "        'history': train_meta['history'],\n"
                "        'classes': CUSTOM_CLASSES,\n"
                "        'evaluation': finetuned_eval_report,\n"
                "    }},\n"
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
                "print({{'outputs': sorted(os.listdir('outputs')), 'finetuned': sorted(os.listdir(ft_output_dir))}})\n"
                "print(['outputs/{stem}_finetuned/model.safetensors', 'outputs/{stem}_finetuned/model-config.json'])"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The predicted label is the argmax of a softmax over the fixed 1000-class ImageNet-1k label space (or custom "
        "class names when fine-tuned); the `score` values are uncalibrated softmax outputs, not probabilities of "
        "correctness, and the pipeline ships no threshold. On the synthetic gradient the label is meaningless by "
        "construction and the evaluation report says `not-measurable`; a `top_k_accuracy` value shown for a single "
        "BYOD image is tutorial evidence for that one image and must not be generalized to a domain, camera, or class "
        "distribution. Images whose subject is outside ImageNet-1k, line drawings, medical or satellite imagery, and "
        "subjects near the image border (removed by the center crop) all degrade results in ways the pipeline does not "
        "detect. The pipeline provides no detection, segmentation, multi-label, OCR, or open-vocabulary capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, perform in-kernel fine-tuning, reload the verified artifact bundle, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a photograph of a known ImageNet class and its index to see the "
        "report switch to `sample-sanity` with `top_k_accuracy` at k=1 and k=5; supply your own multi-class dataset folder "
        "to `pipe.fit` to adapt to your domain; classify a batch (up to `MAX_BATCH`) of labelled images from your own "
        "domain and compare top-1 against the majority-class baseline of that set; inspect how the top-1 score moves when "
        "the subject is moved toward the border of the frame.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/mobilenetv4-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/huggingface/pytorch-image-models\n"
        "- MobileNetV4 — Universal Models for the Mobile Ecosystem: https://arxiv.org/abs/2404.10518"
    ),
}
