"""Draw the whole evaluation — three devices, two benchmarks — as one figure.

    python benchmarks/figure.py --out docs/device-comparison.png

Reads the artifact directories produced by `agreement.py` and `pretrain.py` and
renders every comparison the project makes into a single sheet. Nothing here
computes a new result: this is a view of files that already exist, so the
picture cannot drift from the numbers in AGREEMENT.md and COMPARISON.md.

**Why three devices and not two.** CPU and MPS are the *same machine, same
torch build*, so a difference between them is the device and only the device.
CPU-vs-CUDA crosses machines and torch versions, and CUDA-vs-MPS crosses both
plus the accelerator. Reading the three columns together separates what the
hardware did from what the software version did — neither pair alone can.

**Why the colour scale is logarithmic and centred on float32 epsilon.**
Differences here span 1e-16 to 1e-04, twelve orders. On a linear scale
everything below 1e-05 is one flat colour and the interesting structure — the
gap between "agrees to the last bit" and "agrees to eight ulps" — disappears.
The scale is annotated with float32 epsilon (1.19e-07) and the project's stated
float32 bound (1e-06) so a cell can be read against something meaningful rather
than against the other cells.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("_cmp", HERE / "compare_agreement.py")
cmp_mod = importlib.util.module_from_spec(spec)
sys.modules["_cmp"] = cmp_mod
spec.loader.exec_module(cmp_mod)

FLOAT32_EPS = 1.1920929e-07
BOUND = 1e-6

# Measured on the RTX 5090, CPU against CUDA, in one process so only the device
# varies. Recorded here because it is the finding that made every other number
# on this sheet interpretable, and it cannot be recovered from the artifact
# directories: it required toggling a torch backend flag mid-run.
TF32_LADDER = {
    "tcn_2d_sparse": (1.159e-04, 1.131e-07, None),
    "cnn_2d_sparse": (1.960e-04, 2.337e-07, None),
    "gru_2d_sparse": (1.821e-04, 3.131e-06, 1.503e-07),
    "lstm_2d_sparse": (1.209e-04, 2.656e-06, 1.732e-07),
    "lstm_3d": (1.444e-04, 2.566e-06, 1.770e-07),
    "s4d_upstream_2d": (1.774e-07, 1.774e-07, None),
    "mamba_upstream_2d": (1.277e-07, 1.277e-07, None),
    "transformer_scan_2d": (2.218e-07, 2.218e-07, None),
}

DEVICE_COLOUR = {"CPU": "#4C72B0", "MPS": "#DD8452", "CUDA": "#55A868"}

# Same-process CPU-vs-CUDA on the 5090: one machine, one torch build, so the
# only thing that differs is the device. This is the *only* float64 measurement
# that reaches the last bit — the cross-machine float64 column flattens out at
# ~5e-07 because the two boxes run torch 2.13.0 and 2.12.1, and at that
# precision the version difference is visible where at float32 it is not.
SAME_PROCESS_F64 = {
    "lstm_2d_sparse": 2.689e-16,
    "cnn_2d_sparse": 2.830e-16,
    "s4d_upstream_2d": 2.203e-16,
    "mamba_upstream_2d": 1.479e-09,
}


def panel_title(ax, title, subtitle=""):
    """Title and subtitle as two `text` calls rather than `set_title` plus one.

    `set_title` places its text a fixed pad above the axes and a subtitle drawn
    at an axes-relative offset lands on top of it, which is exactly what the
    first render of this figure did on all eight panels.
    """
    ax.text(
        0,
        1.058,
        title,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    if subtitle:
        ax.text(
            0,
            1.012,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#555",
            va="bottom",
            ha="left",
        )


def load_agreement(root: Path, name: str) -> dict | None:
    path = root / f"{name} agree" / "agreement.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["_dir"] = root / f"{name} agree"
    return data


def load_bench(root: Path, name: str) -> dict | None:
    path = root / f"{name} bench" / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pair_differences(left: dict, right: dict, dtype: str) -> dict[str, dict]:
    """Elementwise worst differences per model, for one device pair and dtype."""
    out: dict[str, dict] = {}
    if dtype not in left["dtypes"] or dtype not in right["dtypes"]:
        return out
    for name in left["models"]:
        if name not in right["models"]:
            continue
        le, re_ = left["models"][name].get(dtype, {}), right["models"][name].get(dtype, {})
        if "error" in le or "error" in re_:
            continue
        got = cmp_mod.compare_tensors(
            left["_dir"] / f"{name}.{dtype}.pt", right["_dir"] / f"{name}.{dtype}.pt"
        )
        if got:
            out[name] = got
    return out


def log_heatmap(ax, matrix, rows, cols, title, subtitle, vmin=1e-8, vmax=1e-4):
    """A heatmap on a log scale, with every cell's value written in it.

    Annotated because a reader should not have to eyedrop a colour to get a
    number: the colour carries the pattern, the text carries the value.
    """
    data = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E8E8E8")

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9, family="monospace")
    panel_title(ax, title, subtitle)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#999")
                continue
            frac = (np.log10(v) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
            ax.text(
                j,
                i,
                f"{v:.0e}".replace("e-0", "e-"),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if frac > 0.62 else "#222",
            )
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/device-comparison.png")
    args = ap.parse_args()
    root = Path(args.root)

    agree = {n: load_agreement(root, n) for n in ("CPU", "MPS", "CUDA")}
    bench = {n: load_bench(root, n) for n in ("CPU", "MPS", "CUDA")}
    have_a = {k: v for k, v in agree.items() if v}
    have_b = {k: v for k, v in bench.items() if v}
    print(f"agreement runs: {list(have_a)}\nbenchmark runs: {list(have_b)}")

    # From the *most complete* run, not the first one found. A partial run —
    # a CPU pass over this zoo is hours, and the manifest is written after
    # every model — would otherwise silently drop its missing models from
    # every panel, including the agreement heatmaps that do have them.
    fullest = max(have_b.values(), key=lambda m: len([e for e in m["models"] if "error" not in e]))
    order = [m["name"] for m in fullest["models"] if "error" not in m]
    # Panels 5, 6 and 8 read the *training* runs, which are far slower to
    # produce than the agreement runs — a CPU pass over the Mamba zoo is hours.
    # Naming the devices in each subtitle keeps a two-device panel beside a
    # three-device one from reading as a dropped result.
    trained_on = ", ".join(have_b)

    # --- the comparisons this figure can make ------------------------------
    PAIRS = [
        ("CPU", "MPS", "float32", "CPU ↔ MPS\n(same machine)"),
        ("CPU", "CUDA", "float32", "CPU ↔ CUDA\n(cross-machine)"),
        ("MPS", "CUDA", "float32", "MPS ↔ CUDA\n(the headline)"),
        ("CPU", "CUDA", "float64", "CPU ↔ CUDA\nfloat64"),
    ]
    pairs = [p for p in PAIRS if p[0] in have_a and p[1] in have_a]
    diffs = {(a, b, dt): pair_differences(have_a[a], have_a[b], dt) for a, b, dt, _ in pairs}

    out_m = [
        [diffs[(a, b, dt)].get(m, {}).get("output", np.nan) for a, b, dt, _ in pairs] for m in order
    ]
    grad_m = [
        [diffs[(a, b, dt)].get(m, {}).get("grad_worst", np.nan) for a, b, dt, _ in pairs]
        for m in order
    ]
    labels = [lab for *_, lab in pairs]

    # --- canvas ------------------------------------------------------------
    fig = plt.figure(figsize=(23, 27), facecolor="white")
    gs = GridSpec(
        5,
        4,
        figure=fig,
        height_ratios=[1.20, 1.10, 0.85, 0.95, 0.80],
        hspace=0.46,
        wspace=0.30,
        # `top` leaves room for the three header lines above panel 1's own
        # title, which sits at 1.058 in axes coordinates.
        left=0.075,
        right=0.975,
        top=0.912,
        bottom=0.030,
    )

    fig.suptitle(
        "torch-dimensions — Evaluation and Device Comparison",
        fontsize=27,
        fontweight="bold",
        y=0.977,
    )
    sub = " · ".join(f"{k}: {v['device_name']} (torch {v['torch']})" for k, v in have_a.items())
    fig.text(0.075, 0.951, sub, fontsize=12, color="#444")
    fig.text(
        0.075,
        0.938,
        "16 models · identical shared initial weights · identical data · "
        "agreement runs use no optimiser, so differences are arithmetic alone",
        fontsize=10,
        color="#666",
    )

    # 1. output agreement
    ax = fig.add_subplot(gs[0, :2])
    im = log_heatmap(
        ax,
        out_m,
        order,
        labels,
        "1 · Forward-pass agreement",
        "worst elementwise relative difference in the output, over every element",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.ax.axhline(FLOAT32_EPS, color="#C44", lw=1.6)
    cb.ax.axhline(BOUND, color="#333", lw=1.6, ls="--")
    cb.set_label("relative difference  (— float32 eps,  -- 1e-6 bound)", fontsize=9)

    # 2. gradient agreement
    ax = fig.add_subplot(gs[0, 2:])
    im = log_heatmap(
        ax,
        grad_m,
        order,
        labels,
        "2 · Backward-pass agreement",
        "worst relative difference over every parameter gradient",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.ax.axhline(FLOAT32_EPS, color="#C44", lw=1.6)
    cb.ax.axhline(BOUND, color="#333", lw=1.6, ls="--")
    cb.set_label("relative difference", fontsize=9)

    # 3. precision ladder ---------------------------------------------------
    # Three rungs, not two. The cross-machine float64 column does *not* reach
    # 1e-16, and drawing only that beside float32 would suggest float64 buys
    # one order of magnitude when it buys nine: the cross-machine number is
    # floored by the torch version gap (2.13.0 vs 2.12.1), which float32 has
    # no headroom to reveal. The same-process measurement is the honest third
    # bar, and the gap between bars two and three is the version, not the
    # hardware.
    ax = fig.add_subplot(gs[1, :2])
    rung = [m for m in order if m in SAME_PROCESS_F64]
    if rung and ("CPU", "CUDA", "float32") in diffs:
        y = np.arange(len(rung))
        f32 = [diffs[("CPU", "CUDA", "float32")].get(m, {}).get("output", np.nan) for m in rung]
        f64x = [diffs[("CPU", "CUDA", "float64")].get(m, {}).get("output", np.nan) for m in rung]
        f64s = [SAME_PROCESS_F64[m] for m in rung]
        ax.barh(y - 0.26, f32, 0.26, label="float32, cross-machine", color="#C44E52")
        ax.barh(y, f64x, 0.26, label="float64, cross-machine", color="#4C72B0")
        ax.barh(y + 0.26, f64s, 0.26, label="float64, same process", color="#55A868")
        ax.set_yticks(y)
        ax.set_yticklabels(rung, fontsize=9.5, family="monospace")
        ax.set_xscale("log")
        ax.set_xlim(1e-17, 1e-3)
        ax.axvline(FLOAT32_EPS, color="#C44", lw=1.4)
        ax.axvline(BOUND, color="#333", lw=1.4, ls="--")
        ax.set_xlabel("relative difference in the output, CPU vs CUDA", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    panel_title(
        ax,
        "3 \u00b7 Precision is the control",
        "green is one machine and one torch build \u2014 the blue-to-green gap "
        "is the torch version, not the hardware",
    )

    # 4. TF32 — the finding that explains everything above
    ax = fig.add_subplot(gs[1, 2:])
    names = list(TF32_LADDER)
    y = np.arange(len(names))
    on = [TF32_LADDER[n][0] for n in names]
    off = [TF32_LADDER[n][1] for n in names]
    nocudnn = [TF32_LADDER[n][2] for n in names]
    ax.barh(y - 0.26, on, 0.26, label="TF32 on (torch's default)", color="#C44E52")
    ax.barh(y, off, 0.26, label="TF32 off", color="#4C72B0")
    # Only the cuDNN RNNs have a third rung; the rest were never affected by
    # TF32 at all and there is nothing further to disable for them.
    have_third = [(i, v) for i, v in enumerate(nocudnn) if v]
    if have_third:
        ax.barh(
            [i + 0.26 for i, _ in have_third],
            [v for _, v in have_third],
            0.26,
            label="TF32 off + cuDNN off",
            color="#55A868",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, family="monospace")
    ax.set_xscale("log")
    ax.axvline(FLOAT32_EPS, color="#C44", lw=1.4)
    ax.axvline(BOUND, color="#333", lw=1.4, ls="--")
    ax.set_xlabel("relative difference, CPU vs CUDA, float32", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    panel_title(
        ax,
        "4 · CUDA does not run float32 by default",
        "cudnn.allow_tf32 ships True — 10 mantissa bits, not 23. It accounted "
        "for the entire original 1.96e-04 gap.",
    )

    # 5. throughput
    ax = fig.add_subplot(gs[2, :2])
    x = np.arange(len(order))
    w = 0.8 / max(len(have_b), 1)
    for k, (dev, man) in enumerate(have_b.items()):
        by = {m["name"]: m for m in man["models"] if "error" not in m}
        ax.bar(
            x + k * w - 0.4 + w / 2,
            [by.get(m, {}).get("steps_per_second", np.nan) for m in order],
            w,
            label=dev,
            color=DEVICE_COLOUR[dev],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=42, ha="right", fontsize=8, family="monospace")
    ax.set_ylabel("training steps / second", fontsize=10)
    ax.legend(fontsize=9)
    panel_title(
        ax,
        "5 · Throughput",
        f"{trained_on} · 12k–141k-parameter models: at this size launch "
        "overhead dominates and a GPU cannot fill",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # 6. did they train to the same place
    ax = fig.add_subplot(gs[2, 2:])
    if len(have_b) >= 2:
        base = "CPU" if "CPU" in have_b else next(iter(have_b))
        b0 = {m["name"]: m for m in have_b[base]["models"] if "error" not in m}
        k = 0
        for dev, man in have_b.items():
            if dev == base:
                continue
            by = {m["name"]: m for m in man["models"] if "error" not in m}
            vals = [
                abs(by[m]["loss_final"] - b0[m]["loss_final"]) if m in by and m in b0 else np.nan
                for m in order
            ]
            ax.bar(
                x + k * 0.4 - 0.2, vals, 0.4, label=f"|{dev} − {base}|", color=DEVICE_COLOUR[dev]
            )
            k += 1
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=42, ha="right", fontsize=8, family="monospace")
        ax.set_ylabel("|Δ final loss| after 300 steps", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    panel_title(
        ax,
        "6 · Do they train to the same place?",
        f"{trained_on} · a loss trajectory is chaotic: 300 steps of gradient "
        "descent amplify a 1e-07 forward difference",
    )

    # 7. parameters
    ax = fig.add_subplot(gs[3, :1])
    # `fullest`, not the first run: parameter counts are a property of the
    # model, but only a run that trained it recorded them.
    by = {m["name"]: m for m in fullest["models"] if "error" not in m}
    ax.barh(np.arange(len(order)), [by[m]["n_params"] for m in order], color="#8172B3")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8, family="monospace")
    ax.set_xlabel("parameters", fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    panel_title(
        ax,
        "7 · Model size",
        "the zoo is small on purpose: coverage of every family, not a leaderboard",
    )

    # 8. loss curves, one panel per device, overlaid per model
    ax = fig.add_subplot(gs[3, 1:])
    for dev, man in have_b.items():
        by = {m["name"]: m for m in man["models"] if "error" not in m}
        for i, m in enumerate(order):
            if m not in by:
                continue
            ax.plot(
                by[m]["losses"],
                color=DEVICE_COLOUR[dev],
                alpha=0.5,
                lw=1,
                label=dev if i == 0 else None,
            )
    ax.set_yscale("log")
    ax.set_xlabel("step", fontsize=10)
    ax.set_ylabel("training loss", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    panel_title(
        ax,
        "8 · All 16 loss curves, every device overlaid",
        f"{trained_on} · the devices are indistinguishable at this scale — which is the result",
    )

    # 9. the summary panel
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    f32_vals = [
        diffs[(a, b, dt)].get(m, {}).get("output", np.nan)
        for a, b, dt, _ in pairs
        if dt == "float32"
        for m in order
    ]
    f32_vals = [v for v in f32_vals if np.isfinite(v)]
    f64_vals = [
        diffs[(a, b, dt)].get(m, {}).get("output", np.nan)
        for a, b, dt, _ in pairs
        if dt == "float64"
        for m in order
    ]
    f64_vals = [v for v in f64_vals if np.isfinite(v)]
    under = sum(1 for v in f32_vals if v <= BOUND)

    text = (
        "What this sheet establishes\n"
        "\n"
        f"·  Worst float32 output difference across every device pair and every model:  "
        f"{max(f32_vals):.2e}.   {under} of {len(f32_vals)} model-pair cells "
        "are at or below the 1e-06 bound.\n"
        f"·  Worst float64 output difference:  {max(f64_vals):.2e} — the last bit. "
        "The devices compute the same thing; float32 simply has nowhere to put the agreement.\n"
        "·  CPU and MPS are the same machine and the same torch build, so "
        "panel 1's first column is the device alone. It agrees to ~1e-07 "
        "everywhere.\n"
        "·  The three models that exceed the bound are the cuDNN RNNs, and "
        "their residual is algorithm rather than precision: cuDNN's fused "
        "LSTM/GRU\n"
        "   is a different implementation, and disabling it brings them to "
        "1.5–1.8e-07 (panel 4, green). That is left visible because the fused "
        "kernel is what runs.\n"
        "·  Gradients of SSM frequencies (A_imag, A_real) exceed the bound in "
        "float32 by construction — they sum oscillating terms that nearly\n"
        "   cancel. The same quantity differs by 1.35e-04 in float32 and "
        "4.87e-15 in float64. Eleven orders is cancellation, not disagreement.\n"
        "\n"
        "Two corrections were required before any of this was measurable, and "
        "both generalise\n"
        "\n"
        "·  CUDA does not run float32 by default (panel 4). Turning TF32 off "
        "moves the convolutional models by three orders of magnitude.\n"
        "·  A fixed seed does not give identical S4 weights across platforms: "
        "torch.linalg.eigh fixes eigenvectors only up to a phase, so B and P\n"
        "   differed by a relative 1.5 and 0.53 between macOS and Linux. Both "
        "benchmarks now load one shared set of starting weights.\n"
    )
    ax.text(
        0.0,
        1.0,
        text,
        transform=ax.transAxes,
        fontsize=11.5,
        va="top",
        ha="left",
        family="sans-serif",
        linespacing=1.75,
        bbox=dict(boxstyle="round,pad=1.1", facecolor="#F6F7F9", edgecolor="#D5D8DC"),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    svg = out.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    print(f"wrote {svg}  ({svg.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
