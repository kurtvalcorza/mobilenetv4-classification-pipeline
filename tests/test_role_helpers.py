"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from mobilenetv4_classification_pipeline import (
    INPUT_SCHEMA,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    NUM_CLASSES,
    evaluation_report,
    validate_inputs,
)


def _image(side: int = 32) -> Image.Image:
    return Image.new("RGB", (side, side), (10, 20, 30))


def _result(indices: list[int]) -> dict:
    return {
        "predictions": [
            {
                "predicted_index": indices[0],
                "predicted_label": "x",
                "top_k": [{"label": "x", "index": i, "score": 0.1} for i in indices],
            }
        ],
        "top_k": len(indices),
        "decision_rule": "argmax",
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(48)], top_k=3, names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [1, MAX_IMAGE_SIDE]
    assert manifest["schema"]["batch"] == [1, MAX_BATCH]
    assert manifest["schema"]["top_k"] == [1, NUM_CLASSES]
    assert manifest["inputs"] == [
        {"id": "a", "mode": "RGB", "size": [32, 32]},
        {"id": "b", "mode": "RGB", "size": [48, 48]},
    ]
    assert manifest["top_k"] == 3
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_image_default_ids() -> None:
    manifest = validate_inputs(_image())
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]


def test_validate_inputs_rejects_like_predict() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 8)))
    with pytest.raises(ValueError, match="MAX_BATCH"):
        validate_inputs([_image()] * (MAX_BATCH + 1))
    with pytest.raises(TypeError):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs([_image()], names=["a", "b"])


def test_evaluation_report_not_measurable_without_targets() -> None:
    report = evaluation_report(_result([3, 7, 9, 1, 2]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert "ImageNet-1k class indices" in report["needs"]
    assert report["n_predictions"] == 1
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_targets() -> None:
    report = evaluation_report(_result([3, 7, 9, 1, 2]), [7], sample_kind="BYOD")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD"
    metrics = {(m["id"], m["k"]): m["value"] for m in report["metrics"]}
    assert metrics == {("top_k_accuracy", 1): 0.0, ("top_k_accuracy", 5): 1.0}
    assert all(m["estimation"] for m in report["metrics"])


def test_evaluation_report_k_bounded_by_result_top_k() -> None:
    report = evaluation_report(_result([3, 7]), [3])
    assert [m["k"] for m in report["metrics"]] == [1, 2]
