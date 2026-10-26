"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/mobilenetv4_classification_colab.ipynb"
BYOD_CELL = "mobilenetv4_classification-12"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


DATASET_CELL = "mobilenetv4_classification-23"
STEM = "mobilenetv4_classification"
EARLIER = {"image": "old", "result": "old", "report": "old", "fine_tuned_pipe": "old", "val_accuracy": "old", "payload": "old"}


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_finetuned").mkdir(parents=True)
    for name in ("input_manifest.json", "evaluation_report.json", "result.json", "top_k.csv", "finetuned_evaluation_report.json"):
        (out / f"{STEM}_{name}").write_text("{}", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = (source or _cell(BYOD_CELL)).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns: dict = {"NUM_CLASSES": 1000, **EARLIER}
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    return {
        "text file (the scripted /etc/hostname run)": (str(tmp_path / "hostname"), "hostname", "Pillow cannot decode this file"),
        "folder": (str(tmp_path / "folder"), "folder", "does not exist or is not a file"),
        "missing path": (str(tmp_path / "nope.jpg"), "nope.jpg", "does not exist or is not a file"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path"])
def test_image_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    path, file_name, rule = _cases(tmp_path)[case]
    ns = {}
    with pytest.raises(ValueError) as caught:
        ns = _section4(monkeypatch, tmp_path, path)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert "run this cell again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert not ns
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_image_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'notes.txt'", 1)
    ns: dict = {"NUM_CLASSES": 1000, "pipe": "kept", **EARLIER}
    with pytest.raises(ValueError, match="^notes.txt: Pillow cannot decode"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0 files"), ({"a.png": b"1", "b.png": b"2"}, "a.png, b.png", "received 2 files")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: Upload exactly one image file") and rule in line


def test_image_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (32, 32)).save(tmp_path / "ok.png")
    source = _cell(BYOD_CELL).replace('    """BYOD_PATH first', "    raise RuntimeError('internal bug')\n    \"\"\"BYOD_PATH first", 1)
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "ok.png"), source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_good_image_still_loads(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (32, 24)).save(tmp_path / "ok.png")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "ok.png"))
    assert ns["image_name"] == "ok.png" and ns["image"].size == (32, 24) and ns["sample_kind"] == "BYOD"


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a text file present in every fresh runtime."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: Pillow cannot decode this file")


# --- Section 8 dataset --------------------------------------------------------------------------------------------


def _png(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _section8(monkeypatch, tmp_path, path: str, source: str | None = None) -> dict:
    """Section 4 on its default path (it defines the DAT20 helpers), then Section 8 up to training."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {"NUM_CLASSES": 1000}
    exec(compile(_cell(BYOD_CELL), "<section 4>", "exec"), ns)
    _seed_outputs(tmp_path)
    ns.update(EARLIER)
    source = source or _cell(DATASET_CELL)
    source = source[: source.index("ft_output_dir = ")].replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1).replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {path!r}", 1)
    exec(compile(source, "<section 8>", "exec"), ns)
    return ns


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "one.zip").write_bytes(_zip({"cats/a.png": _png(1), "cats/b.png": _png(2)}))
    (tmp_path / "text.zip").write_bytes(_zip({"notes.txt": b"x"}))
    return {
        "not a .zip": ("hostname", "expected a dataset .zip archive ending in .zip"),
        "corrupt zip": ("fake.zip", "not a readable .zip archive"),
        "one class": ("one.zip", "at least 2 distinct classes"),
        "no images": ("text.zip", "No supported image files"),
        "missing": ("nope.zip", "does not exist or is not a file"),
    }


@pytest.mark.parametrize("case", ["not a .zip", "corrupt zip", "one class", "no images", "missing"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_training_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _dataset_cases(tmp_path)[case]
    holder = {}

    def run():
        holder["ns"] = _section8(monkeypatch, tmp_path, str(tmp_path / file_name))

    with pytest.raises(ValueError) as caught:
        run()
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert "earlier results of this session and their outputs/ files were removed" in line
    remaining = sorted(p.name for p in (tmp_path / "outputs").iterdir())
    assert remaining == [f"{STEM}_evaluation_report.json", f"{STEM}_input_manifest.json"], remaining  # Sections 4-6 belong to the image


def test_dataset_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "two.zip").write_bytes(_zip({"cats/a.png": _png(1), "cats/b.png": _png(2), "dogs/c.png": _png(3), "dogs/d.png": _png(4)}))
    source = _cell(DATASET_CELL).replace("            def _open(member):\n", "            def _open(member):\n                raise RuntimeError('internal bug')\n", 1)
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section8(monkeypatch, tmp_path, str(tmp_path / "two.zip"), source)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_dataset_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    """The pinned sample is not user input: its digest mismatch keeps the plain ValueError and its traceback."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {"NUM_CLASSES": 1000}
    exec(compile(_cell(BYOD_CELL), "<section 4>", "exec"), ns)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"tampered"

    request = types.SimpleNamespace(Request=lambda *a, **k: None, urlopen=lambda *a, **k: Response())
    ns["urllib"] = types.SimpleNamespace(request=request)
    source = _cell(DATASET_CELL)
    source = source[: source.index("ft_output_dir = ")].replace("import urllib.request\n", "")
    with pytest.raises(ValueError, match="SHA-256 mismatch") as caught:
        exec(compile(source, "<section 8>", "exec"), ns)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_good_dataset_still_splits(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "two.zip").write_bytes(_zip({"cats/a.png": _png(1), "cats/b.png": _png(2), "dogs/c.png": _png(3), "dogs/d.png": _png(4)}))
    ns = _section8(monkeypatch, tmp_path, str(tmp_path / "two.zip"))
    assert ns["CUSTOM_CLASSES"] == ["cats", "dogs"] and ns["dataset_source"] == "BYOD: two.zip"
