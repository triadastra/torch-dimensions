"""Check our sequential axial sweep against upstream S4ND — the real thing.

    python dossier/verify_s4nd.py

`tests/test_published_composition.py` checks the same identity against a
*transcription* of S4ND's composition rule, so that CI can run it anywhere
with no vendored code. This script is the stronger and less portable version:
it imports **state-spaces/s4 itself**, takes the kernels its own code
produces, and asks whether our one-axis-per-layer sweep reproduces its
simultaneous N-D FFT.

The upstream repo is not vendored. Clone it wherever you like and point
`TD_EXTERNAL` at the parent directory; nothing here redistributes it.

**Why it runs off-GPU at all.** Two things stood between this repo and a Mac,
and neither was the mathematics:

1. `src/utils/train.py` imports `pytorch_lightning` at module scope, for a
   logger that no forward pass touches. Stubbed in `_shims.py`.
2. The Cauchy/Vandermonde CUDA extension is absent, so the kernel falls back
   to its pure-torch path — which upstream prints a warning about and which is
   exactly what we want.

`hydra-core` and `einops` do have to be installed; they are genuine imports,
not packaging accidents. They are dev-only here and deliberately not in this
project's dependencies.

**What the comparison holds fixed.** Upstream is constructed with
`linear=True`, `channels=1`, `bidirectional=False`, so its forward reduces to
the joint convolution plus the `D` skip and nothing else — no FF, no
activation, no dropout. We then apply *its* per-axis kernels through our
`axial_apply`, one axis at a time, and add the same `D` term. Any difference
is therefore composition, not parameterization.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _shims import external, stub_lightning  # noqa: E402

sys.path.insert(0, str(external("s4")))
stub_lightning()

from src.models.sequence.modules.s4nd import S4ND  # noqa: E402

import torch_dimensions as td  # noqa: E402


def causal_fft_conv(seq: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """`(M, A, H)` in and out — the bare convolution, no skip, no gate."""
    a = seq.shape[1]
    n = 2 * a
    xt = seq.transpose(1, 2)
    y = torch.fft.irfft(torch.fft.rfft(xt, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :a]
    return y.transpose(1, 2)


def compare(device: str, shape: tuple[int, ...], d_model: int = 4, seed: int = 0) -> float:
    torch.manual_seed(seed)
    # contract_version=0 is hardcoded to two axes upstream — it indexes
    # k_f[0], k_f[1] literally. Version 1 builds the einsum string
    # programmatically and is the one that generalizes. That this choice
    # exists, and that the default does not survive rank 3, is the "hardcoded
    # to one rank" problem this library was written about, visible in the
    # source of the paper that introduced N-D SSMs.
    upstream = (
        S4ND(
            d_model=d_model,
            d_state=16,
            l_max=shape,
            dim=len(shape),
            bidirectional=False,
            linear=True,
            transposed=False,
            mode="diag",
            contract_version=0 if len(shape) == 2 else 1,
        )
        .eval()
        .to(device)
    )

    x = torch.randn(2, *shape, d_model, device=device)
    with torch.no_grad():
        theirs = upstream(x)
        theirs = theirs[0] if isinstance(theirs, tuple) else theirs

        kernels = [
            kern(L=length)[0].squeeze(0)
            for kern, length in zip(upstream.kernel, shape, strict=True)
        ]
        names = tuple("hwd"[: len(shape)])
        lat = td.Lattice(shape=shape, names=names)
        ours = x
        for name, k in zip(names, kernels, strict=True):
            ours = td.axial_apply(ours, lat, name, lambda s, k=k: causal_fft_conv(s, k))
        ours = ours + x * upstream.D.squeeze(0)  # the same skip term

    diff = (ours - theirs).abs().max().item()
    print(
        f"  {device:4s} rank {len(shape)} {str(shape):10s} "
        f"max |ours - upstream| = {diff:.3e}   (output scale {theirs.abs().max().item():.3f})"
    )
    return diff


def main() -> None:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    if torch.cuda.is_available():
        devices.append("cuda")
    print("upstream S4ND's simultaneous N-D kernel vs our sequential sweep")
    print("(upstream's own kernels, upstream's own D skip)\n")
    worst = 0.0
    for device in devices:
        for shape in [(6, 7), (4, 5, 6)]:
            worst = max(worst, compare(device, shape))
    print(f"\nworst case across all runs: {worst:.3e}")


if __name__ == "__main__":
    main()
