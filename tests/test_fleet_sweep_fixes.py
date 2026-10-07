"""Regression tests for the 2026-10-05 fleet-sweep fixes of the tutorial notebook (SWP-R restart guard, SWP-G guided
layer, SWP-B BYOD path). They need only CI's dependencies: the notebook's own kernel cell is executed with a stand-in
IPython shell, and the isolated worker it starts runs on this interpreter (a stand-in for the managed CPython), so
the routing protocol, environment reuse and Section 1 idempotence are exercised for real without any download."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook_sweep", TOOLS / "build_notebook.py")
TEMPLATE = _load("notebook_template_sweep", TOOLS / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


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


def _kernel_cell(nb: dict) -> dict:
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "# dimer: kernel cell" in _src(c)]
    assert len(cells) == 1, "exactly one kernel (bootstrap) cell"
    return cells[0]


def _cell_with(nb: dict, marker: str) -> str:
    found = [s for s in _code_cells(nb) if marker in s]
    assert len(found) == 1, f"exactly one code cell must contain {marker!r}"
    return found[0]


def _function(source: str, name: str, namespace: dict) -> object:
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"<{name}>", "exec"), namespace)
    return namespace[name]


# --- SWP-R: restart guard replaced by the isolated runtime -------------------------------------------------------


def test_swp_r_no_pip_install_or_restart_in_any_cell(nb: dict) -> None:
    """SWP-R: nothing is pip-installed into the kernel and no cell asks for a restart."""
    code = "\n".join(_code_cells(nb))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in code
    assert "packages_distributions" not in code


def test_swp_r_lock_is_carried_hash_locked_and_matches_pins(nb: dict) -> None:
    """SWP-R: the carried lock is the committed lock, every pin is in it at the same version, every entry hashed."""
    kernel = _src(_kernel_cell(nb))
    lock_text = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    digest = hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_SHA256 = {digest!r}" in kernel
    build.check_lock(build.load_context(ROOT, TEMPLATE)["pins"], lock_text)
    assert "'--require-hashes', '--only-binary', ':all:'" in kernel


def test_swp_r_environment_keyed_on_lock_and_child_env_cleaned(nb: dict) -> None:
    """SWP-R: the environment folder is keyed on the lock digest and reused; the worker gets MPLBACKEND=Agg and no
    PYTHONPATH/PYTHONHOME/PYTHONSTARTUP."""
    kernel = _src(_kernel_cell(nb))
    assert "'dimer_isolated_env_' + LOCK_SHA256[:12]" in kernel
    assert "elif _isolated_environment_ready():" in kernel
    assert 'MPLBACKEND="Agg"' in kernel
    assert '("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"' in kernel
    meta = _kernel_cell(nb)["metadata"]
    assert meta.get("cellView") == "form"


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def _run_kernel_cell(source: str, namespace: dict, shell: _Shell) -> str:
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    display_mod = types.ModuleType("IPython.display")
    display_mod.display = lambda *a, **k: None
    saved = {k: sys.modules.get(k) for k in ("IPython", "IPython.display")}
    sys.modules["IPython"], sys.modules["IPython.display"] = ipython, display_mod
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(source, "<kernel cell>", "exec"), namespace)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return out.getvalue()


@pytest.mark.skipif(sys.platform != "linux", reason="the worker protocol uses Linux pass_fds")
def test_swp_r_section1_reuses_environment_and_worker_when_rerun(nb: dict, tmp_path: Path, monkeypatch) -> None:
    """SWP-R: with a complete environment for this lock present, Section 1 builds nothing (no download), starts one
    worker, routes cells to it, and a second run of Section 1 keeps the same worker and its variables."""
    kernel = _src(_kernel_cell(nb))
    digest = re.search(r"LOCK_SHA256 = '([0-9a-f]{64})'", kernel).group(1)
    env = tmp_path / ("dimer_isolated_env_" + digest[:12])
    (env / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env / "bin" / "python")  # stand-in for the managed CPython
    (env / ".dimer-lock-sha256").write_text(digest + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("a ready environment must not download"))
    shell = _Shell()
    namespace: dict = {"__name__": "__main__"}
    first = _run_kernel_cell(kernel, namespace, shell)
    assert "'reused': True" in first and "'worker_reused': False" in first
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert len(shell.input_transformers_cleanup) == 1
        routed = shell.input_transformers_cleanup[0](["value = 41\n"])
        assert routed == ["_DIMER_ISOLATED_RUNTIME.run('value = 41\\n')\n"]
        assert shell.input_transformers_cleanup[0](["# dimer: kernel cell\nx = 1\n"]) == ["# dimer: kernel cell\nx = 1\n"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runtime.run("value = 41\nimport os\nprint(value + 1, os.environ.get('MPLBACKEND'), 'PYTHONSTARTUP' in os.environ)")
        assert out.getvalue().split() == ["42", "Agg", "False"]
        second = _run_kernel_cell(kernel, namespace, shell)
        assert "'worker_reused': True" in second
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and len(shell.input_transformers_cleanup) == 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runtime.run("print(value)")
        assert out.getvalue().strip() == "41", "re-running Section 1 must not strand later cells"
        with pytest.raises(namespace["IsolatedCellError"]), contextlib.redirect_stderr(io.StringIO()):
            runtime.run("raise ValueError('boom')")
    finally:
        runtime.close()


# --- SWP-G: guided layer ------------------------------------------------------------------------------------------

GUIDED_MARKERS = (
    "**Who this notebook is for.**",
    "**How to use this notebook.**",
    "**Roadmap:**",
    "## Troubleshooting",
    "## Glossary",
    "## Conclusion",
)


def test_swp_g_guided_layer_present(nb: dict) -> None:
    """SWP-G: audience, how-to-use, roadmap, predictions, checkpoints, troubleshooting, glossary, conclusion."""
    md = _markdown(nb)
    missing = [m for m in GUIDED_MARKERS if m not in md]
    assert not missing, missing
    assert md.count("**Predict before running:**") >= 5
    assert md.count("<summary>Check your reasoning</summary>") >= 5


def test_swp_g_infrastructure_cells_labelled_and_collapsed(nb: dict) -> None:
    """SWP-G (GDL11): setup, carried-module and snapshot cells are labelled Infrastructure and collapsed."""
    md = _markdown(nb)
    assert md.count("> **Infrastructure.**") >= 3
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = _src(cell)
        if "# dimer: kernel cell" in s or cell["metadata"].get("dimer", {}).get("embedded_module") or "MANIFEST = {" in s:
            assert cell["metadata"].get("cellView") == "form", s[:80]


def test_swp_g_no_leftover_placeholders(nb: dict) -> None:
    """SWP-G: no literal template placeholders reach the learner."""
    learner_code = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_module")]
    text = _markdown(nb) + "\n".join(learner_code)
    for token in ("{{MODEL_ID}}", "{MODEL_ID}", "{MODEL_REVISION}", "{stem}", "@P:"):
        assert token not in text, token
    assert "{{" not in _markdown(nb)


# --- SWP-B: BYOD works from a path on any runtime, refusals name the file and the rule ---------------------------


def _byod_reader(nb: dict) -> object:
    from pathlib import Path as _Path

    return _function(_cell_with(nb, "BYOD_PATH = ''"), "read_byod_file", {"os": os, "Path": _Path})


def test_swp_b_byod_path_fields_default_off(nb: dict) -> None:
    """SWP-B: BYOD_PATH / BYOD_DATASET_PATH form fields exist; both gates stay off by default."""
    code = "\n".join(_code_cells(nb))
    assert "BYOD_PATH = ''  # @param" in code and "BYOD_DATASET_PATH = ''  # @param" in code
    assert "USE_BYOD = False  # @param" in code and "USE_BYOD_DATASET = False  # @param" in code


def test_swp_b_byod_path_reads_file_and_refuses_with_names(nb: dict, tmp_path: Path) -> None:
    """SWP-B: a path works without Colab; a missing path, a wrong suffix and an empty path off Colab are named."""
    read = _byod_reader(nb)
    image = tmp_path / "leaf.png"
    image.write_bytes(b"\x89PNG data")
    assert read(str(image), "image file") == ("leaf.png", b"\x89PNG data")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        read(str(tmp_path / "missing.png"), "image file")
    with pytest.raises(ValueError, match=r"leaf.png: expected a dataset .zip archive ending in \.zip"):
        read(str(image), "dataset .zip archive", (".zip",))
    sys.modules.pop("google.colab", None)
    with pytest.raises(RuntimeError, match="only in Google Colab"):
        read("", "image file")


def test_swp_b_cancelled_colab_upload_gives_a_clear_message(nb: dict, monkeypatch) -> None:
    """SWP-B: a cancelled (empty) Colab upload stops with 'Upload exactly one', not StopIteration."""
    read = _byod_reader(nb)
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: {}
    colab.files = files
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match="Upload exactly one image file"):
        read("", "image file")
    files.upload = lambda: {"a.png": b"1"}
    assert read("", "image file") == ("a.png", b"1")


def _dataset_prefix(nb: dict) -> str:
    source = _cell_with(nb, "USE_BYOD_DATASET = False")
    prefix = source.split("ft_output_dir = ")[0]
    return prefix.replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1)


def _zip(entries: dict[str, bytes]) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


_PNG_COUNTER = [0]


def _png() -> bytes:
    """A distinct 8x8 PNG on every call (Section 8 removes pixel-identical duplicates before splitting)."""
    from PIL import Image

    _PNG_COUNTER[0] += 1
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (_PNG_COUNTER[0] % 256, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _run_dataset(nb: dict, archive: bytes) -> dict:
    import numpy as np
    from PIL import Image

    namespace = {"np": np, "Image": Image, "SAMPLE_SIDE": 256, "read_byod_file": lambda *a, **k: ("mine.zip", archive)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_dataset_prefix(nb), "<section 8>", "exec"), namespace)
    return namespace


def test_swp_b_dataset_split_counts_and_reports_skipped_members(nb: dict) -> None:
    """SWP-B: the true minimum (2 classes x 2 images) is accepted, unused members are counted, not dropped silently."""
    ns = _run_dataset(nb, _zip({"cats/a.png": _png(), "cats/b.png": _png(), "dogs/c.png": _png(), "dogs/d.png": _png(), "notes.txt": b"x"}))
    assert ns["CUSTOM_CLASSES"] == ["cats", "dogs"]
    assert (len(ns["train_images"]), len(ns["val_images"])) == (2, 2)
    assert ns["skipped"] == {"not_an_image": 1, "val_class_not_in_train": 0, "pixel_duplicate": 0}
    ns = _run_dataset(nb, _zip({"train/a/1.png": _png(), "train/b/2.png": _png(), "val/a/3.png": _png(), "val/c/4.png": _png()}))
    assert ns["skipped"]["val_class_not_in_train"] == 1 and len(ns["val_images"]) == 1


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ({"cats/a.png": b"P", "dogs/c.png": b"P"}, "fewer than 2 images"),
        ({"cats/a.png": b"P", "cats/b.png": b"P"}, "at least 2 distinct classes"),
        ({"cats/a.png": b"P", "cats/b.png": b"broken", "dogs/c.png": b"P", "dogs/d.png": b"P"}, "Pillow cannot decode"),
        ({"train/a/1.png": b"P", "train/b/2.png": b"P", "val/c/3.png": b"P"}, "val/ holds no image"),
        ({"readme.txt": b"x"}, "No supported image files"),
    ],
)
def test_swp_b_dataset_refusals_name_the_rule(nb: dict, entries: dict, match: str) -> None:
    """SWP-B: each refusal is a ValueError naming the class, member or rule (no KeyError/StopIteration/NameError)."""
    archive = _zip({k: (_png() if v == b"P" else v) for k, v in entries.items()})
    with pytest.raises(ValueError, match=match):
        _run_dataset(nb, archive)


def test_swp_b_synthetic_fallback_has_no_undefined_names(nb: dict) -> None:
    """SWP-B: the offline fallback dataset uses SAMPLE_SIDE from Section 4 (it used undefined SAMPLE_HEIGHT/WIDTH)."""
    source = _cell_with(nb, "USE_BYOD_DATASET = False")
    assert "SAMPLE_HEIGHT" not in source and "SAMPLE_WIDTH" not in source
    assert "np.zeros((SAMPLE_SIDE, SAMPLE_SIDE, 3)" in source
