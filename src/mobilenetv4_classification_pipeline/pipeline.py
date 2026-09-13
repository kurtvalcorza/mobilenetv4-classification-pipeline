"""ImageNet-1k classification with the pinned ``timm/mobilenetv4_conv_small.e2400_r224_in1k`` checkpoint.

MobileNetV4-Conv-Small: 3.8 M parameters, ImageNet-1k, 224 px; the edge/CPU profile of the DIMER timm set.
The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or,
when explicitly allowed, from the Hugging Face Hub at the pinned revision. Preprocessing is
the upstream ``pretrained_cfg`` (resize/crop/normalize) resolved through ``timm.data``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "timm/mobilenetv4_conv_small.e2400_r224_in1k"
MODEL_REVISION = "331fb803779522b685cf942e15f914fb6741c1eb"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "mobilenetv4-conv-small"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

NUM_CLASSES = 1000
MAX_IMAGE_SIDE = 4096  # pixels; larger images are rejected before any decode-to-tensor work
MAX_BATCH = 64  # images per predict() call
DEFAULT_TOP_K = 5
DECISION_RULE = "argmax"  # the label reported as `predicted_index` is the softmax argmax; no threshold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _hub_reference(model_id: str, revision: str) -> str:
    """timm's ``hf-hub:owner/name@revision`` form; ``hf_split`` passes ``revision=`` to hf_hub_download."""
    return f"hf-hub:{model_id}@{revision}"


def top_k_accuracy(predictions: Sequence[Any], targets: Sequence[int], k: int = 1) -> float:
    """Fraction of items whose target index is among the first ``k`` predicted indices.

    ``predictions`` may be the per-image dicts returned by ``predict`` or plain index sequences.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    hits = 0
    for pred, target in zip(predictions, targets, strict=True):
        ranked = pred["top_k"] if isinstance(pred, Mapping) else pred
        indices = [int(item["index"]) if isinstance(item, Mapping) else int(item) for item in ranked]
        hits += int(target in indices[:k])
    return hits / len(predictions)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image or a sequence of them; any mode, converted to RGB",
    "image_side_px": [1, MAX_IMAGE_SIDE],
    "batch": [1, MAX_BATCH],
    "top_k": [1, NUM_CLASSES],
    "preprocessing": "resize 256 px, center-crop 224x224 (crop_pct 0.875, bicubic), ImageNet normalisation",
}


def _check_inputs(images: Any, top_k: int, max_classes: int = NUM_CLASSES) -> list[Image.Image]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the images as a list."""
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, Sequence) or isinstance(images, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if not 1 <= len(images) <= MAX_BATCH:
        raise ValueError(f"batch size must be between 1 and MAX_BATCH={MAX_BATCH}, got {len(images)}")
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
        width, height = image.size
        if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
            raise ValueError(f"image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if not 1 <= top_k <= max_classes:
        raise ValueError(f"top_k must be between 1 and {max_classes}")
    return list(images)


def validate_inputs(
    images: Image.Image | Sequence[Image.Image],
    top_k: int = DEFAULT_TOP_K,
    *,
    names: Sequence[str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``predict`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    checked = _check_inputs(images, top_k, max_classes=num_classes)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size)}
            for i, image in enumerate(checked)
        ],
        "top_k": top_k,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], targets: Sequence[int] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``targets`` (one ImageNet-1k index per prediction) the report carries ``top_k_accuracy``
    at k=1 and k=5 as sample-sanity evidence; without them the verdict is ``not-measurable`` and
    the report says what labelled data would make the task measurable.
    """
    predictions = result["predictions"]
    base = {
        "task": "imagenet-1k single-label classification",
        "decision_rule": result.get("decision_rule", DECISION_RULE),
        "sample_kind": sample_kind,
        "n_predictions": len(predictions),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if targets is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth class index was supplied for the evaluated images",
            "needs": (
                "labelled photographs with ImageNet-1k class indices (0-999), e.g. a held-out sample of your "
                "own data, scored with top_k_accuracy against the majority-class baseline of that sample"
            ),
        }
    top_k = int(result.get("top_k", DEFAULT_TOP_K))
    ks = sorted({1, min(5, top_k)})
    return {
        **base,
        "metrics": [
            {
                "id": "top_k_accuracy",
                "k": k,
                "value": top_k_accuracy(predictions, list(targets), k=k),
                "estimation": "single sample, no dispersion estimate",
            }
            for k in ks
        ],
        "verdict": "sample-sanity",
        "reason": f"{len(predictions)} labelled image(s) from the tutorial sample; not a benchmark",
        "needs": "a labelled evaluation set from the deployment domain for any generalisable accuracy claim",
    }


@dataclass
class MobileNetV4ClassificationPipeline:
    """``_runner`` maps a float tensor (N, 3, H, W) to logits (N, NUM_CLASSES); injectable for tests."""

    _runner: Callable[[Any], Any]
    _transform: Callable[[Image.Image], Any]
    device: str = "cpu"
    labels: tuple[str, ...] = ()
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> MobileNetV4ClassificationPipeline:
        import timm
        import torch
        from timm.data import ImageNetInfo, create_transform, resolve_model_data_config

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        arch_name = MODEL_ID.split("/", 1)[1]
        ft_config_path = root / "model-config.json"
        if ft_config_path.is_file() and (root / WEIGHTS_FILE).is_file():
            with open(ft_config_path, encoding="utf-8") as fh:
                ft_config = json.load(fh)
            num_classes = ft_config.get("num_classes", len(ft_config.get("class_names", [])))
            model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)
            from safetensors.torch import load_file

            state_dict = load_file(root / WEIGHTS_FILE)
            model.load_state_dict(state_dict, strict=True)
            source = "finetuned-local"
            labels = tuple(ft_config.get("class_names", [str(i) for i in range(num_classes)]))
        elif (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / CONFIG_FILE, encoding="utf-8") as fh:
                config = json.load(fh)
            snapshot_name = f"{config['architecture']}.{config['pretrained_cfg']['tag']}"
            if snapshot_name != arch_name:
                raise ValueError(f"snapshot config names {snapshot_name!r}, expected {arch_name!r}")
            overlay = dict(config["pretrained_cfg"])
            overlay["file"] = str(root / WEIGHTS_FILE)  # 'file' takes precedence over hf_hub_id in timm
            model = timm.create_model(
                arch_name, pretrained=True, pretrained_cfg_overlay=overlay, num_classes=NUM_CLASSES
            )
            source = "local-snapshot"
            info = ImageNetInfo(subset="imagenet-1k")
            labels = tuple(info.index_to_description(i) for i in range(info.num_classes()))
        elif allow_download:
            model = timm.create_model(
                _hub_reference(MODEL_ID, revision=MODEL_REVISION), pretrained=True, num_classes=NUM_CLASSES
            )
            source = "hf-hub"
            info = ImageNetInfo(subset="imagenet-1k")
            labels = tuple(info.index_to_description(i) for i in range(info.num_classes()))
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(resolved_device)
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        return cls(runner, transform, resolved_device, labels, source)

    def fit(
        self,
        train_images: Sequence[Image.Image],
        train_targets: Sequence[int],
        val_images: Sequence[Image.Image] | None = None,
        val_targets: Sequence[int] | None = None,
        class_names: Sequence[str] | None = None,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> tuple[MobileNetV4ClassificationPipeline, dict[str, Any]]:
        """Fine-tune the classification head in-kernel with AdamW and cross-entropy loss."""
        import timm
        import torch
        from safetensors.torch import load_file, save_file
        from timm.data import create_transform, resolve_model_data_config

        if len(train_images) != len(train_targets):
            raise ValueError("train_images and train_targets must have the same length")
        if not train_images:
            raise ValueError("train_images must not be empty")

        unique_targets = sorted(set(train_targets))
        num_classes = len(class_names) if class_names is not None else len(unique_targets)
        labels = tuple(class_names) if class_names is not None else tuple(str(i) for i in range(num_classes))

        arch_name = MODEL_ID.split("/", 1)[1]
        model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        weights_path = root / WEIGHTS_FILE
        if weights_path.is_file():
            base_state = load_file(weights_path)
            backbone_weights = {
                k: v for k, v in base_state.items()
                if not k.startswith("classifier.")
            }
            model.load_state_dict(backbone_weights, strict=False)

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device)

        data_config = resolve_model_data_config(model)
        train_transform = create_transform(**data_config, is_training=True)
        eval_transform = create_transform(**data_config, is_training=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        history: list[dict[str, Any]] = []

        for epoch in range(epochs):
            model.train()
            indices = list(range(len(train_images)))
            running_loss = 0.0
            total_samples = 0

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_tensors = torch.stack(
                    [train_transform(train_images[j].convert("RGB")) for j in batch_idx]
                ).to(resolved_device)
                batch_labels = torch.tensor(
                    [train_targets[j] for j in batch_idx], dtype=torch.long, device=resolved_device
                )

                optimizer.zero_grad()
                logits = model(batch_tensors)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_idx)
                total_samples += len(batch_idx)

            train_loss = running_loss / total_samples if total_samples > 0 else 0.0
            epoch_record: dict[str, Any] = {"epoch": epoch + 1, "train_loss": round(train_loss, 5)}

            if val_images and val_targets and len(val_images) == len(val_targets):
                model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.inference_mode():
                    for v_i in range(0, len(val_images), batch_size):
                        v_batch_imgs = val_images[v_i : v_i + batch_size]
                        v_batch_tgts = val_targets[v_i : v_i + batch_size]
                        v_tensors = torch.stack(
                            [eval_transform(img.convert("RGB")) for img in v_batch_imgs]
                        ).to(resolved_device)
                        v_labels = torch.tensor(v_batch_tgts, dtype=torch.long, device=resolved_device)
                        v_logits = model(v_tensors)
                        v_loss = criterion(v_logits, v_labels)
                        val_running_loss += v_loss.item() * len(v_batch_imgs)
                        preds = torch.argmax(v_logits, dim=-1)
                        val_correct += (preds == v_labels).sum().item()
                        val_total += len(v_batch_imgs)

                epoch_record["val_loss"] = round(val_running_loss / val_total, 5) if val_total > 0 else 0.0
                epoch_record["val_accuracy"] = round(val_correct / val_total, 4) if val_total > 0 else 0.0

            history.append(epoch_record)

        model.eval()

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), out_path / WEIGHTS_FILE)
            config_payload = {
                "architecture": arch_name,
                "num_classes": num_classes,
                "class_names": list(labels),
                "fine_tuned": True,
                "base_model": MODEL_ID,
                "base_revision": MODEL_REVISION,
            }
            with open(out_path / "model-config.json", "w", encoding="utf-8") as fh:
                json.dump(config_payload, fh, indent=2)

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        fitted_pipe = MobileNetV4ClassificationPipeline(
            runner,
            eval_transform,
            resolved_device,
            labels,
            source="fine-tuned",
        )
        return fitted_pipe, {"history": history, "num_classes": num_classes, "class_names": list(labels)}

    def _validate(self, images: Any, top_k: int) -> list[Image.Image]:
        max_classes = len(self.labels) if self.labels else NUM_CLASSES
        return _check_inputs(images, top_k, max_classes=max_classes)

    def predict(
        self, images: Image.Image | Sequence[Image.Image], top_k: int | None = None
    ) -> dict[str, Any]:
        """Classify images; ``score`` is a softmax score over classes, not a calibrated probability."""
        import torch

        active_classes = len(self.labels) if self.labels else NUM_CLASSES
        resolved_top_k = min(DEFAULT_TOP_K, active_classes) if top_k is None else top_k
        batch_images = self._validate(images, resolved_top_k)
        batch = torch.stack([self._transform(image.convert("RGB")) for image in batch_images])
        logits = self._runner(batch)
        expected_classes = active_classes
        if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch_images), expected_classes):
            raise RuntimeError(f"runner must return a tensor of shape (batch, {expected_classes})")
        scores = torch.softmax(logits.float(), dim=-1).cpu()
        values, indices = torch.topk(scores, k=resolved_top_k, dim=-1)
        predictions = []
        for image_values, image_indices in zip(values.tolist(), indices.tolist(), strict=True):
            image_values = [float(s) for s in image_values]
            ranked = [
                {"label": self.labels[i] if i < len(self.labels) else str(i), "index": i, "score": s}
                for s, i in zip(image_values, image_indices, strict=True)
            ]
            best = ranked[0]
            predictions.append(
                {"predicted_index": best["index"], "predicted_label": best["label"], "top_k": ranked}
            )
        return {
            "predictions": predictions,
            "top_k": resolved_top_k,
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
