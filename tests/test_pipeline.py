import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from PIL import Image

from mobilenetv4_classification_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUM_CLASSES,
    MobileNetV4ClassificationPipeline,
    stage_missing_files,
    top_k_accuracy,
    verify_snapshot,
)
from mobilenetv4_classification_pipeline import pipeline as pipeline_module

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fake_transform(image: Image.Image) -> torch.Tensor:
    return torch.zeros(3, 8, 8)


def _fake_runner(batch: torch.Tensor) -> torch.Tensor:
    logits = torch.zeros(batch.shape[0], NUM_CLASSES)
    logits[:, 7] = 5.0  # class 7 wins for every image
    logits[:, 3] = 2.0
    return logits


def _pipeline() -> MobileNetV4ClassificationPipeline:
    labels = tuple(f"class-{i}" for i in range(NUM_CLASSES))
    return MobileNetV4ClassificationPipeline(_fake_runner, _fake_transform, "cpu", labels, "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "timm/mobilenetv4_conv_small.e2400_r224_in1k"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest["files"][0]["sha256"] = flipped
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        MobileNetV4ClassificationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_hub_reference_carries_pinned_revision():
    reference = pipeline_module._hub_reference(MODEL_ID, revision=MODEL_REVISION)
    assert reference == f"hf-hub:{MODEL_ID}@{MODEL_REVISION}"


def test_predict_rejects_bad_inputs():
    pipe = _pipeline()
    image = Image.new("RGB", (32, 32))
    with pytest.raises(TypeError):
        pipe.predict("not-an-image")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.predict([image, 42])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="batch size"):
        pipe.predict([])
    with pytest.raises(ValueError, match="batch size"):
        pipe.predict([image] * (MAX_BATCH + 1))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.predict(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 1)))
    with pytest.raises(ValueError, match="top_k"):
        pipe.predict(image, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        pipe.predict(image, top_k=NUM_CLASSES + 1)
    with pytest.raises(TypeError):
        pipe.predict(image, top_k=2.5)  # type: ignore[arg-type]


def test_predict_output_fields():
    pipe = _pipeline()
    result = pipe.predict([Image.new("L", (16, 16)), Image.new("RGB", (16, 16))], top_k=3)
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    assert result["top_k"] == 3
    assert result["decision_rule"] == "argmax"
    assert len(result["predictions"]) == 2
    first = result["predictions"][0]
    assert first["predicted_index"] == 7
    assert first["predicted_label"] == "class-7"
    assert [item["index"] for item in first["top_k"]][:2] == [7, 3]
    assert set(first["top_k"][0]) == {"label", "index", "score"}
    assert 0.0 < first["top_k"][0]["score"] <= 1.0
    assert sum(item["score"] for item in first["top_k"]) <= 1.0 + 1e-6


def test_top_k_accuracy():
    pipe = _pipeline()
    result = pipe.predict([Image.new("RGB", (16, 16))] * 2, top_k=5)
    assert top_k_accuracy(result["predictions"], [7, 3], k=1) == 0.5
    assert top_k_accuracy(result["predictions"], [7, 3], k=2) == 1.0
    assert top_k_accuracy([[1, 2], [3, 4]], [2, 9], k=2) == 0.5
    with pytest.raises(ValueError):
        top_k_accuracy([[1]], [1, 2])
    with pytest.raises(ValueError):
        top_k_accuracy([], [])


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_predict_fine_tuned_reloaded_artifact_default_top_k(tmp_path: Path) -> None:
    """Regression test: reloaded 2-class artifact on default prediction path bounds top_k to num_classes."""
    import timm
    from safetensors.torch import save_file

    arch_name = MODEL_ID.split("/", 1)[1]
    model = timm.create_model(arch_name, pretrained=False, num_classes=2)
    weights_path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, weights_path)
    config = {
        "num_classes": 2,
        "class_names": ["class_a", "class_b"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    (tmp_path / "model-config.json").write_text(json.dumps(config), encoding="utf-8")

    pipe = MobileNetV4ClassificationPipeline.from_pretrained(weights_dir=tmp_path)
    assert pipe.labels == ("class_a", "class_b")
    assert pipe.source == "finetuned-local"

    # Default prediction path: top_k unspecified -> defaults to min(DEFAULT_TOP_K, 2) == 2
    image = Image.new("RGB", (32, 32), color=(100, 150, 200))
    result = pipe.predict(image)
    assert result["top_k"] == 2
    assert len(result["predictions"]) == 1
    pred = result["predictions"][0]
    assert pred["predicted_label"] in ["class_a", "class_b"]
    assert len(pred["top_k"]) == 2
    assert {item["label"] for item in pred["top_k"]} == {"class_a", "class_b"}

    # Explicit top_k exceeding active classes must be rejected
    with pytest.raises(ValueError, match="top_k"):
        pipe.predict(image, top_k=3)
    with pytest.raises(ValueError, match="top_k"):
        pipe.predict(image, top_k=5)


def test_fit_fails_closed_missing_base_weights(tmp_path: Path) -> None:
    """Regression test: fit() must fail closed with FileNotFoundError if base weights/manifest are missing."""
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    with pytest.raises(FileNotFoundError):
        MobileNetV4ClassificationPipeline.fit(
            train_images=[img, img],
            train_targets=[0, 1],
            val_images=[img, img],
            val_targets=[0, 1],
            class_names=["class_a", "class_b"],
            weights_dir=tmp_path / "nonexistent",
            allow_download=False,
        )


def test_fit_fails_closed_corrupted_base_weights(tmp_path: Path) -> None:
    """Regression test: fit() must fail closed with ValueError if base weights do not match manifest."""
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.safetensors", "bytes": 100, "sha256": "0" * 64},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"corrupted_bytes")

    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    with pytest.raises(ValueError, match="sha256|size"):
        MobileNetV4ClassificationPipeline.fit(
            train_images=[img, img],
            train_targets=[0, 1],
            val_images=[img, img],
            val_targets=[0, 1],
            class_names=["class_a", "class_b"],
            weights_dir=tmp_path,
            allow_download=False,
        )


