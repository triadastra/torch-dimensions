"""Say what precision the arithmetic actually ran at, and let it be chosen.

CUDA does not run float32 by default. `torch.backends.cudnn.allow_tf32` ships
as **True**, so every cuDNN convolution and RNN runs in TF32 — 10 mantissa
bits where float32 has 23. (`matmul.allow_tf32` ships as False, which is why
attention, Mamba and S4 were never affected.)

A device comparison that leaves this alone is not comparing devices. It
compares MPS's float32 against CUDA's TF32 and reports the difference as a
property of the hardware. Measured on this zoo, CPU against CUDA on one box:

| model | TF32 on (default) | TF32 off | + cuDNN off |
|---|---|---|---|
| `cnn_2d_sparse` | 1.96e-04 | 2.34e-07 | — |
| `tcn_2d_sparse` | 1.16e-04 | 1.13e-07 | — |
| `lstm_2d_sparse` | 1.21e-04 | 2.66e-06 | 1.73e-07 |
| `gru_2d_sparse` | 1.82e-04 | 3.13e-06 | 1.50e-07 |
| everything else | ~2e-07 | ~2e-07 | — |

So the entire 1e-04 signature was TF32, and turning it off moves the
convolutional models by three orders of magnitude. What is left for the RNNs
is not precision but implementation: cuDNN's fused LSTM/GRU is a different
algorithm, and disabling it brings them to the same ~1.7e-07 floor as
everything else. That is a real difference worth reporting rather than
configuring away, which is why `--cudnn off` exists and is not the default.

The two flags are set separately because they answer different questions:
`--tf32` controls *precision* and belongs off in a numerical comparison and at
torch's own defaults in a training run;
`--cudnn` controls *which implementation runs* and belongs on, because it is
what a user gets.

Both are recorded in every run's JSON. A number whose precision is not written
down cannot be compared with another one later.
"""

from __future__ import annotations

import torch


def add_arguments(ap, *, tf32_default: str) -> None:
    ap.add_argument(
        "--tf32",
        choices=("torch", "on", "off"),
        default=tf32_default,
        help="CUDA TF32. 'off' makes float32 mean float32, which is what a "
        "cross-device numerical comparison needs. 'torch' leaves torch's own "
        "defaults alone (cuDNN on, matmul off) — what a user actually gets. "
        f"'on' forces both. Default here: {tf32_default}.",
    )
    ap.add_argument(
        "--cudnn",
        choices=("on", "off"),
        default="on",
        help="cuDNN's fused kernels. 'off' forces the generic implementation, "
        "which shows whether a residual difference is precision or algorithm. "
        "Slower, and not what a user runs — diagnostic only.",
    )


def apply(args) -> dict:
    """Set the knobs and return what was actually set, for the manifest.

    Returns the *observed* values rather than the requested ones: on a machine
    with no CUDA these settings do nothing, and a manifest claiming "tf32: off"
    on an MPS run would imply a control that was never applied.
    """
    if torch.cuda.is_available():
        # "torch" deliberately touches nothing: torch ships cuDNN TF32 on and
        # matmul TF32 off, and a training run should describe that combination
        # rather than a third one this file invented.
        if args.tf32 != "torch":
            enable = args.tf32 == "on"
            torch.backends.cudnn.allow_tf32 = enable
            torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.enabled = args.cudnn == "on"
        return {
            "tf32_requested": args.tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_enabled": torch.backends.cudnn.enabled,
        }
    return {
        "tf32_requested": args.tf32,
        "cudnn_tf32": None,
        "matmul_tf32": None,
        "cudnn_enabled": None,
        "note": "no CUDA device; these settings are CUDA-only and did nothing",
    }


def describe(settings: dict) -> str:
    if settings.get("cudnn_tf32") is None:
        return "TF32: n/a (no CUDA)"
    return (
        f"TF32: cudnn={settings['cudnn_tf32']} matmul={settings['matmul_tf32']}"
        f" · cuDNN enabled: {settings['cudnn_enabled']}"
    )
