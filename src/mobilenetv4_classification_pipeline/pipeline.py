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
        if (root / MANIFEST_NAME).is_file():
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
        elif allow_download:
            model = timm.create_model(
                _hub_reference(MODEL_ID, revision=MODEL_REVISION), pretrained=True, num_classes=NUM_CLASSES
            )
            source = "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(resolved_device)
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)
        info = ImageNetInfo(subset="imagenet-1k")
        labels = tuple(info.index_to_description(i) for i in range(info.num_classes()))

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        return cls(runner, transform, resolved_device, labels, source)

    def _validate(self, images: Any, top_k: int) -> list[Image.Image]:
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
        if not 1 <= top_k <= NUM_CLASSES:
            raise ValueError(f"top_k must be between 1 and NUM_CLASSES={NUM_CLASSES}")
        return list(images)

    def predict(
        self, images: Image.Image | Sequence[Image.Image], top_k: int = DEFAULT_TOP_K
    ) -> dict[str, Any]:
        """Classify images; ``score`` is a softmax score over 1000 classes, not a calibrated probability."""
        import torch

        batch_images = self._validate(images, top_k)
        batch = torch.stack([self._transform(image.convert("RGB")) for image in batch_images])
        logits = self._runner(batch)
        if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch_images), NUM_CLASSES):
            raise RuntimeError("runner must return a tensor of shape (batch, NUM_CLASSES)")
        scores = torch.softmax(logits.float(), dim=-1).cpu()
        values, indices = torch.topk(scores, k=top_k, dim=-1)
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
            "top_k": top_k,
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
