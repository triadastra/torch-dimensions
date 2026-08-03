"""What precision the benchmarks actually ran at, recorded rather than assumed.

CUDA does not run float32 by default. `torch.backends.cudnn.allow_tf32` ships
as True, so cuDNN convolutions and RNNs execute in TF32 — 10 mantissa bits
against float32's 23. `matmul.allow_tf32` ships as False, which is why the
attention, Mamba and S4 models were never affected and the convolutional and
recurrent ones were.

The device comparison was therefore measuring MPS float32 against CUDA TF32
and attributing the difference to the hardware. With TF32 off, `cnn_2d_sparse`
moves from 1.96e-04 to 2.34e-07 and `tcn_2d_sparse` from 1.16e-04 to 1.13e-07.

These tests run on any machine, including one with no GPU, because the thing
worth pinning is not the GPU behaviour — it is that the setting is applied,
reported honestly when it could not be applied, and never silently assumed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

spec = importlib.util.spec_from_file_location("_td_precision", ROOT / "benchmarks" / "precision.py")
precision = importlib.util.module_from_spec(spec)
sys.modules["_td_precision"] = precision
spec.loader.exec_module(precision)


def parse(argv, *, tf32_default="off"):
    ap = argparse.ArgumentParser()
    precision.add_arguments(ap, tf32_default=tf32_default)
    return ap.parse_args(argv)


def test_the_numerical_benchmark_defaults_to_real_float32():
    """`agreement.py` asks for TF32 off, because comparing one device's float32
    against another's TF32 measures a precision setting and calls it hardware."""
    assert parse([], tf32_default="off").tf32 == "off"


def test_the_training_benchmark_defaults_to_what_users_get():
    """`pretrain.py` keeps torch's own default, since its speed numbers should
    describe the configuration people actually run."""
    assert parse([], tf32_default="on").tf32 == "on"


def test_cudnn_stays_on_unless_asked():
    """Turning cuDNN off changes *which implementation runs*, not its
    precision. It answers a diagnostic question — the RNN residual is the fused
    kernel, not float32 — and must not be the default, because the fused kernel
    is what a user's model uses."""
    assert parse([]).cudnn == "on"
    assert parse(["--cudnn", "off"]).cudnn == "off"


def test_the_settings_are_reported_as_observed_not_as_requested():
    """On a machine with no CUDA these knobs do nothing. A manifest that
    recorded "tf32: off" there would claim a control that was never applied,
    and the whole point of the record is that a number's precision travels
    with it."""
    settings = precision.apply(parse(["--tf32", "off"]))
    assert settings["tf32_requested"] == "off"

    if torch.cuda.is_available():
        assert settings["cudnn_tf32"] is False
        assert settings["matmul_tf32"] is False
    else:
        assert settings["cudnn_tf32"] is None
        assert settings["matmul_tf32"] is None
        assert "no CUDA" in settings["note"]


def test_describe_never_claims_a_control_it_did_not_apply():
    without = {"tf32_requested": "off", "cudnn_tf32": None}
    assert "n/a" in precision.describe(without)

    with_cuda = {
        "tf32_requested": "off",
        "cudnn_tf32": False,
        "matmul_tf32": False,
        "cudnn_enabled": True,
    }
    text = precision.describe(with_cuda)
    assert "cudnn=False" in text and "matmul=False" in text


def test_applying_the_setting_does_not_disturb_a_cpu_run():
    """The knobs are CUDA-only; asking for them on CPU must not raise or leave
    torch in a state the rest of the suite then inherits."""
    before = torch.backends.cudnn.enabled
    precision.apply(parse(["--tf32", "off"]))
    precision.apply(parse(["--tf32", "on"]))
    if not torch.cuda.is_available():
        assert torch.backends.cudnn.enabled == before


def test_both_benchmarks_record_precision_in_their_manifest():
    """The record is the deliverable: two runs are only comparable if each says
    what precision it ran at, and neither script may quietly drop it."""
    for name in ("agreement.py", "pretrain.py"):
        source = (ROOT / "benchmarks" / name).read_text()
        assert "precision.apply(args)" in source, f"{name} never applies the setting"
        assert '"precision": prec' in source, f"{name} never records the setting"
