"""Regression tests for the 2026-10-02 notebook-review fixes of tutorials/mobilenetv4_classification_colab.ipynb
(MNV-B1 the default adaptation learns and the verdict is computed; MNV-M2 zero-shot reference; MNV-M3 head-only,
seeded, recorded; MNV-m1 duplicate grouping; MNV-m2 fail-closed digest and labelled fallback; MNV-m3 new-data
inference; MNV-m5 reload equivalence; MNV-S2/S3/S4).

They need only CI's dependencies (torch CPU, timm). The learning test runs `fit` on a RANDOMLY INITIALISED
MobileNetV4-Conv-Small (timm.create_model is monkeypatched to ignore the pinned weights) on a synthetic two-class set,
so it is evidence that the training loop moves a zero-initialised head below chance loss and is deterministic — it is
NOT evidence about the pretrained checkpoint, which only a hosted run can give."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import re
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mobilenetv4_classification_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mobilenetv4_classification_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb: dict) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, needle: str) -> str:
    hits = [c for c in _code_cells(nb) if needle in c]
    assert len(hits) == 1, f"exactly one code cell contains {needle!r}, got {len(hits)}"
    return hits[0]


def _defs(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    parts = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(parts) == len(names), f"missing helper(s) among {names}"
    return "\n\n".join(parts)


# --- stand-in model: random init, no pinned weights ------------------------------------------------------------------


def _fake_snapshot(root: Path) -> None:
    """A manifest/config pair that passes fit's identity checks; the weights are never read (create_model is patched)."""
    arch = pl.MODEL_ID.split("/", 1)[1]
    config = {"architecture": arch.split(".")[0], "pretrained_cfg": {"tag": arch.split(".", 1)[1]}}
    config_bytes = json.dumps(config).encode()
    (root / pl.CONFIG_FILE).write_bytes(config_bytes)
    (root / pl.WEIGHTS_FILE).write_bytes(b"stand-in")
    manifest = {
        "modelId": pl.MODEL_ID,
        "revision": pl.MODEL_REVISION,
        "files": [
            {"path": pl.CONFIG_FILE, "bytes": len(config_bytes), "sha256": hashlib.sha256(config_bytes).hexdigest()},
            {"path": pl.WEIGHTS_FILE, "bytes": 8, "sha256": hashlib.sha256(b"stand-in").hexdigest()},
        ],
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def random_init_model(monkeypatch, tmp_path: Path) -> Path:
    """create_model returns a copy of one randomly initialised network whose BatchNorm running statistics were
    calibrated on noise (so its pre-logit features have a sane scale, as a pretrained network's do)."""
    import copy

    import timm
    import torch

    real_create = timm.create_model
    torch.manual_seed(0)
    base = real_create(pl.MODEL_ID.split("/", 1)[1], pretrained=False, num_classes=2).train()
    with torch.no_grad():
        for _ in range(4):
            base(torch.rand(8, 3, 224, 224))
    base.eval()

    def create_model(name, pretrained=False, pretrained_cfg_overlay=None, **kwargs):
        assert kwargs.get("num_classes") == 2
        return copy.deepcopy(base)

    monkeypatch.setattr(timm, "create_model", create_model)
    monkeypatch.setattr(pl, "verify_snapshot", lambda root: {"files": []})
    monkeypatch.setattr(pl, "stage_missing_files", lambda root, allow_download=False: [])
    _fake_snapshot(tmp_path)
    return tmp_path


def _two_class_set(n_per_class: int = 8, seed: int = 0):
    """Horizontal-stripe versus vertical-stripe 64x64 images with per-image colour jitter (trivially separable)."""
    rng = np.random.default_rng(seed)
    images, targets = [], []
    for cls, pattern in enumerate(("horizontal", "vertical")):
        for _ in range(n_per_class):
            arr = np.full((64, 64, 3), rng.integers(0, 60), dtype=np.uint8)
            if pattern == "horizontal":
                arr[::8, :, 0] = 255
            else:
                arr[:, ::8, 1] = 255
            images.append(Image.fromarray(arr))
            targets.append(cls)
    return images, targets


# --- MNV-B1 / MNV-M3: the loop learns, is seeded and records its recipe -----------------------------------------------


def test_mnv_b1_default_recipe_moves_the_head_below_chance_loss_on_a_stand_in(random_init_model: Path, tmp_path: Path) -> None:
    """MNV-B1 (loop half, stand-in): a zero-initialised head starts at ln(2) and the loss falls under the default
    recipe (head-only, lr 1e-3, shuffled, 8 epochs); the old recipe (random head, all parameters, 1e-4, 1 epoch,
    batch 4, BatchNorm in train mode) starts confidently wrong. Random features are not discriminative, so held-out
    accuracy is not asserted here: that is the hosted run's job."""
    train_images, train_targets = _two_class_set(8)
    val_images, val_targets = _two_class_set(3, seed=1)
    out = tmp_path / "ft"
    fitted, meta = pl.MobileNetV4ClassificationPipeline.fit(
        train_images, train_targets, val_images, val_targets, ["horizontal", "vertical"], weights_dir=random_init_model, output_dir=out, device="cpu"
    )
    history = meta["history"]
    assert len(history) == 8 and meta["fine_tuning"]["steps"] == 16  # 16 images / batch 8 x 8 epochs
    assert history[0]["train_loss"] <= math.log(2) + 0.15, "the zero head starts at chance loss (epoch average), never confidently wrong"
    assert history[-1]["train_loss"] < history[0]["train_loss"] - 0.1, f"the head must move: {history}"
    assert all("val_accuracy" in entry for entry in history)
    recipe = meta["fine_tuning"]
    assert recipe["trainable"] == "head" and recipe["head_init"] == "zeros" and recipe["seed"] == 42 and recipe["shuffle"] is True
    assert recipe["trainable_parameters"] == 2 * 1280 + 2 and recipe["frozen_parameters"] + recipe["trainable_parameters"] == recipe["total_parameters"]
    assert recipe["batchnorm_mode"].startswith("eval")
    config = json.loads((out / "model-config.json").read_text(encoding="utf-8"))
    assert config["fine_tuning"] == recipe
    # the previous default: every parameter, lr 1e-4, one epoch of batch-4 steps
    _, old = pl.MobileNetV4ClassificationPipeline.fit(
        train_images, train_targets, val_images, val_targets, ["horizontal", "vertical"], epochs=1, batch_size=4, learning_rate=1e-4, trainable="all", head_init="default", weights_dir=random_init_model, device="cpu"
    )
    assert old["fine_tuning"]["trainable_parameters"] == old["fine_tuning"]["total_parameters"] and old["fine_tuning"]["batchnorm_mode"] == "train"
    assert old["history"][0]["train_loss"] > math.log(2) + 0.5, "the random head starts confidently wrong (the review's 2.4-5.0 regime)"


def test_mnv_m3_two_fits_are_identical_and_the_backbone_is_untouched(random_init_model: Path) -> None:
    """MNV-M3 (acceptance): the same seed gives identical held-out predictions; head-only leaves the backbone equal."""
    import torch

    train_images, train_targets = _two_class_set(6)
    val_images, _ = _two_class_set(3, seed=2)
    kwargs = dict(class_names=["horizontal", "vertical"], epochs=2, weights_dir=random_init_model, device="cpu")
    a, _ = pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, **kwargs)
    b, _ = pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, **kwargs)
    scores_a = [t["score"] for t in a.predict(val_images)["predictions"][0]["top_k"]]
    scores_b = [t["score"] for t in b.predict(val_images)["predictions"][0]["top_k"]]
    assert scores_a == scores_b
    c, _ = pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, seed=7, **kwargs)
    assert [t["score"] for t in c.predict(val_images)["predictions"][0]["top_k"]] != scores_a
    # head-only: every backbone tensor is bit-identical to the start, only the classifier moved
    import timm

    start = timm.create_model("x", num_classes=2).state_dict()
    out = random_init_model / "ft-m3"
    pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, output_dir=out, **kwargs)
    from safetensors.torch import load_file

    trained = load_file(str(out / pl.WEIGHTS_FILE))
    moved = [k for k in start if not torch.equal(start[k], trained[k])]
    assert moved and all(k.startswith("classifier.") for k in moved), moved


def test_mnv_m3_fit_refuses_bad_arguments(random_init_model: Path) -> None:
    train_images, train_targets = _two_class_set(2)
    with pytest.raises(ValueError, match="trainable must be"):
        pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, class_names=["a", "b"], trainable="backbone", weights_dir=random_init_model)
    with pytest.raises(ValueError, match="head_init must be"):
        pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, class_names=["a", "b"], head_init="ones", weights_dir=random_init_model)
    with pytest.raises(ValueError, match="at least 1"):
        pl.MobileNetV4ClassificationPipeline.fit(train_images, train_targets, class_names=["a", "b"], epochs=0, weights_dir=random_init_model)


# --- MNV-B1 / MNV-M2 / MNV-m5 / MNV-S3: Section 9 helpers and static checks ----------------------------------------------


def _section9_helpers(nb: dict) -> dict:
    ns: dict = {"math": math, "NUM_CLASSES": pl.NUM_CLASSES}
    exec(_defs(_cell_with(nb, "def compute_verdict("), {"wilson_interval", "compute_verdict", "zero_shot_predict"}), ns)
    return ns


def test_mnv_b1_verdict_is_computed_not_hard_coded(nb: dict) -> None:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    assert "'verdict': 'success'" not in raw and '"verdict": "success"' not in raw
    h = _section9_helpers(nb)
    assert h["compute_verdict"](0.1667, 0.5, 1.0) == ("below-majority-baseline (the adapted model is worse than guessing the majority class)", "fine-tuned model BELOW the zero-shot reference (adaptation lost ground)")
    assert h["compute_verdict"](0.5, 0.5, None)[0].startswith("at-majority-baseline") and "unavailable" in h["compute_verdict"](0.5, 0.5, None)[1]
    assert h["compute_verdict"](0.95, 0.5, 0.95) == ("above-majority-baseline", "fine-tuned model equal to the zero-shot reference (adaptation added nothing the pretrained model lacked)")
    assert h["compute_verdict"](0.9, 0.5, 0.8)[1].startswith("fine-tuned model above")


def test_mnv_s3_wilson_interval(nb: dict) -> None:
    h = _section9_helpers(nb)
    lo, hi = h["wilson_interval"](20, 40)
    assert lo == pytest.approx(0.352, abs=1e-3) and hi == pytest.approx(0.648, abs=1e-3)
    assert h["wilson_interval"](40, 40)[1] == 1.0 and h["wilson_interval"](0, 0) == (0.0, 0.0)


def test_mnv_m2_zero_shot_reference_reads_the_pretrained_scores_through_groups(nb: dict) -> None:
    h = _section9_helpers(nb)

    class Pipe:
        def predict(self, img, top_k=None):
            assert top_k == pl.NUM_CLASSES
            scores = {i: 0.0 for i in range(pl.NUM_CLASSES)}
            scores[31] = 0.4
            scores[555] = 0.3
            scores[867] = 0.2
            return {"predictions": [{"top_k": [{"index": i, "score": s} for i, s in scores.items()]}]}

    label, totals = h["zero_shot_predict"](Pipe(), None, {"frog": [30, 31, 32], "truck": [555, 569, 675, 717, 864, 867]}, ["frog", "truck"])
    assert label == "truck" and totals == {"frog": 0.4, "truck": pytest.approx(0.5)}
    md = _markdown(nb)
    assert "zero-shot reference" in md.lower() and "`frog` and `truck` are ImageNet-1k classes" in md


def test_mnv_m5_reload_equivalence_is_checked_with_a_tolerance(nb: dict) -> None:
    cell = _cell_with(nb, "RELOAD_TOLERANCE = 1e-5")
    assert "max_reload_diff = max(max_reload_diff" in cell
    assert "raise RuntimeError(f'the reloaded artifact does not reproduce the fine-tuned model" in cell
    assert "print(f'reload equivalence: max |score difference|" in cell


def test_mnv_m3_new_data_inference_and_held_out_table(nb: dict) -> None:
    cell = _cell_with(nb, "RELOAD_TOLERANCE = 1e-5")
    assert "new_data = reloaded_pipe.predict(image)" in cell and "'new_data_inference'" in cell
    assert "held-out images (first 12)" in cell
    assert "EXTRA_SEEDS = 0  # @param" in cell and "'seed_spread'" in cell


# --- MNV-m1 / MNV-m2: Section 8 dataset handling ------------------------------------------------------------------


def _png(colour: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buf, format="PNG")
    return buf.getvalue()


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def _dataset_prefix(nb: dict) -> str:
    source = _cell_with(nb, "USE_BYOD_DATASET = False")
    return source[: source.index("ft_output_dir = ")]


def _run_section8(nb: dict, *, byod_zip: bytes | None = None, urlopen=None) -> tuple[dict, str]:
    import urllib.request

    ns: dict = {"np": np, "Image": Image, "SAMPLE_SIDE": 256, "hashlib": hashlib}
    prefix = _dataset_prefix(nb)
    if byod_zip is not None:
        prefix = prefix.replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1)
        ns["read_byod_file"] = lambda *a, **k: ("mine.zip", byod_zip)
    fake_request = types.SimpleNamespace(Request=urllib.request.Request, urlopen=urlopen)
    ns["urllib"] = types.SimpleNamespace(request=fake_request)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(prefix.replace("import urllib.request\n", ""), "<section 8>", "exec"), ns)
    return ns, out.getvalue()


def test_mnv_m1_pixel_identical_copies_are_grouped_before_the_split(nb: dict) -> None:
    """MNV-m1 (acceptance): with every image shipped twice, no held-out image has a copy in training."""
    entries = {}
    for cls, base in (("frog", 10), ("truck", 120)):
        for i in range(10):
            png = _png((base + i, 5, 5))
            entries[f"CIFAR-10-subset/original_images/{cls}/image_{i}.png"] = png
            entries[f"CIFAR-10-subset/darkened_images/{cls}/image_{i}.png"] = png
    ns, out = _run_section8(nb, byod_zip=_zip(entries))
    assert ns["skipped"]["pixel_duplicate"] == 20
    assert (len(ns["train_images"]), len(ns["val_images"])) == (16, 4)
    train_digests = {hashlib.sha256(np.asarray(img).tobytes()).hexdigest() for img in ns["train_images"]}
    val_digests = {hashlib.sha256(np.asarray(img).tobytes()).hexdigest() for img in ns["val_images"]}
    assert not (train_digests & val_digests)
    assert "'split_assumption': 'images are independent after exact-duplicate removal'" in out
    assert "pixel-identical" in _markdown(nb)


def test_mnv_m2_tampered_sample_archive_fails_closed(nb: dict) -> None:
    """MNV-m2 (acceptance): a digest mismatch raises naming the digests; nothing trains and no fallback runs."""
    tampered = _zip({"frog/a.png": _png((1, 1, 1))})

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return tampered

    with pytest.raises(ValueError, match="SHA-256 mismatch: expected 66f90a4f.*nothing is trained"):
        _run_section8(nb, urlopen=lambda req, timeout=30: Resp())


def test_mnv_m2_offline_fallback_is_labelled(nb: dict) -> None:
    """MNV-m2 (acceptance): with no network the cell runs the synthetic fallback and labels it."""

    def urlopen(req, timeout=30):
        raise OSError("simulated: network unreachable")

    ns, out = _run_section8(nb, urlopen=urlopen)
    assert ns["dataset_fallback_used"] is True and ns["dataset_source"].startswith("synthetic stripes fallback")
    assert ns["CUSTOM_CLASSES"] == ["synthetic_horizontal_stripe", "synthetic_vertical_stripe"]
    assert (len(ns["train_images"]), len(ns["val_images"])) == (8, 4)
    assert "falling back to deterministic synthetic stripes" in out
    cell = _cell_with(nb, "RELOAD_TOLERANCE = 1e-5")
    assert "'dataset_fallback_used': dataset_fallback_used" in cell
    export = _cell_with(nb, "'prediction': result,")
    assert "'dataset_fallback_used': dataset_fallback_used" in export


def test_mnv_b1_section_8_uses_the_new_recipe_and_prints_counts(nb: dict) -> None:
    cell = _cell_with(nb, "USE_BYOD_DATASET = False")
    for field in ("EPOCHS = 8  # @param", "LEARNING_RATE = 1e-3  # @param", "TRAINABLE = 'head'  # @param", "BATCH_SIZE = 8  # @param", "SUBSET_PER_CLASS = 100"):
        assert field in cell, field
    assert "trainable=TRAINABLE" in cell and "seed=SEED" in cell
    assert "'loss_fell_below_chance': final_train_loss < chance_loss" in cell
    assert "'trainable_parameters': fine_tuning_meta['trainable_parameters']" in cell


def test_mnv_b1_prose_tells_the_learner_what_to_expect_and_how_to_read_a_zero_lift(nb: dict) -> None:
    md = _markdown(nb)
    assert "**Expected result (no hosted run of this configuration is recorded yet)" in md
    assert "**How to read a lift at or below zero:**" in md
    assert "**Expected ordering" in md
    assert "**Change one thing (optional):** in Section 8 set `TRAINABLE = \"all\"`" in md
    assert "training only a new head" in md and "3.8 M backbone parameters are frozen" in md
    assert "Section 8 downloads one public dataset archive on the default path" in md
    assert "gracefully falls back" not in md


def test_mnv_m4_prerequisites_name_the_dataset_download(nb: dict) -> None:
    prereq = next(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown" and _src(c).startswith("## Prerequisites"))
    assert "`Cleanlab/cifar-10-subset` (~986 KB" in prereq and "refused on a SHA-256 mismatch" in prereq
    assert "nothing is downloaded" not in prereq


def test_mnv_s4_no_deprecated_fromarray_mode(nb: dict) -> None:
    assert "mode='RGB'" not in "\n".join(_code_cells(nb))


def test_mnv_m1_no_restart_and_every_cell_parses(nb: dict) -> None:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    assert "Restart the runtime" not in raw
    for i, source in enumerate(_code_cells(nb)):
        ast.parse(source, f"cell{i}")
    md = _markdown(nb)
    assert "{{" not in md and re.search(r"\{MODEL_ID\}|@P:", md) is None


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, real_google: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """The isolated worker's google.colab stubs must carry a module spec: on Colab, accelerate/transformers call
    importlib.util.find_spec("google.colab"), which raised `google.colab.__spec__ is None` on a spec-less stub
    (language-model-pipeline T4 run of f2053aa; reference fixes language-model-pipeline d185817 and
    chronos-2-forecasting-pipeline 33a2f53)."""
    import importlib.util
    import os
    import sys

    router = _cell_with(nb, '_WORKER_SOURCE = r"""')
    worker = router[router.index('_WORKER_SOURCE = r"""') + len('_WORKER_SOURCE = r"""') :]
    worker = worker[: worker.index('"""')]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {n: sys.modules[n] for n in names if n in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for n in names:
            sys.modules.pop(n, None)
        # Both branches: no importable `google` (stub created) and an existing namespace package.
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None})
        for n in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(n)  # raised ValueError before the fix
            assert spec is not None and spec.name == n
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
