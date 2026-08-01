"""Dataset fetching for the reproductions. Vendors nothing, pins everything.

Two datasets, both from their canonical public hosts, both checksummed. The
checksums are the point: a reproduction whose inputs are unverified is a
reproduction of whatever the mirror served that day. Files land in
``~/.cache/torch_dimensions`` (override with ``TD_DATA_DIR``) and are
downloaded once.

No torchvision. The library depends on ``torch`` and nothing else, and a
reproduction that quietly needs a second framework is not reproducing with the
library — it is reproducing with the library plus whatever else was lying
around.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import tarfile
from pathlib import Path
from urllib.request import urlopen

import torch

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": (
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
    ),
    "train-labels-idx1-ubyte.gz": (
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
    ),
    "t10k-images-idx3-ubyte.gz": (
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
    ),
}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_SHA = "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd"

# Beijing Multi-Site Air-Quality (UCI 501): 12 monitoring stations x 6
# pollutants x 35,064 hourly steps. Chosen for the sparse-lattice work because
# it is genuinely 2-D — station and pollutant are different kinds of axis, not
# a reshaped sequence — and because its gaps are real measurement gaps.
BEIJING_URL = (
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
)
BEIJING_SHA = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
POLLUTANTS = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")


def data_dir() -> Path:
    root = Path(os.environ.get("TD_DATA_DIR", Path.home() / ".cache" / "torch_dimensions"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch(url: str, dest: Path, sha256: str | None) -> Path:
    """Download once, verify always.

    The checksum is re-checked on every call, not only after a download: a
    truncated or half-written cache file is exactly the failure that presents
    as "the model suddenly stopped learning".
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        with urlopen(url) as r, open(dest, "wb") as fh:  # noqa: S310 — pinned https URLs
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    if sha256:
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != sha256:
            raise RuntimeError(
                f"checksum mismatch for {dest.name}\n  expected {sha256}\n  got      {got}\n"
                "delete the file and retry; if it repeats, the upstream host changed and this "
                "reproduction should not be trusted until the pin is reviewed"
            )
    return dest


def _idx(path: Path) -> torch.Tensor:
    """Read an IDX file (the MNIST container format) into a uint8 tensor."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if raw[0:2] != b"\x00\x00" or raw[2] != 0x08:
        raise ValueError(f"{path.name}: not a uint8 IDX file")
    n_dims = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(n_dims)]
    body = torch.frombuffer(bytearray(raw[4 + 4 * n_dims :]), dtype=torch.uint8)
    return body.reshape(*dims)


def mnist() -> dict[str, torch.Tensor]:
    """MNIST as raw uint8 tensors: images ``(N, 28, 28)``, labels ``(N,)``."""
    root = data_dir() / "mnist"
    for name, sha in MNIST_FILES.items():
        fetch(f"{MNIST_URL}/{name}", root / name, sha)
    return {
        "train_x": _idx(root / "train-images-idx3-ubyte.gz"),
        "train_y": _idx(root / "train-labels-idx1-ubyte.gz"),
        "test_x": _idx(root / "t10k-images-idx3-ubyte.gz"),
        "test_y": _idx(root / "t10k-labels-idx1-ubyte.gz"),
    }


def cifar10() -> dict[str, torch.Tensor]:
    """CIFAR-10 as raw uint8: images ``(N, 32, 32, 3)``, labels ``(N,)``."""
    root = data_dir() / "cifar10"
    archive = fetch(CIFAR_URL, root / "cifar-10-binary.tar.gz", CIFAR_SHA)
    members = {
        "train": [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)],
        "test": ["cifar-10-batches-bin/test_batch.bin"],
    }
    out: dict[str, torch.Tensor] = {}
    with tarfile.open(archive) as tar:
        for split, names in members.items():
            blobs = []
            for name in names:
                fh = tar.extractfile(name)
                if fh is None:
                    raise RuntimeError(f"{archive.name} is missing {name}")
                blobs.append(torch.frombuffer(bytearray(fh.read()), dtype=torch.uint8))
            # Each record is 1 label byte followed by 3072 bytes of plane-major
            # RGB — 1024 red, then green, then blue.
            recs = torch.cat(blobs).reshape(-1, 3073)
            out[f"{split}_y"] = recs[:, 0].clone()
            out[f"{split}_x"] = recs[:, 1:].reshape(-1, 3, 32, 32).permute(0, 2, 3, 1).contiguous()
    return out


def beijing() -> dict:
    """Air quality as a ``(T, station, pollutant)`` tensor plus its axis names.

    Returns the raw readings with gaps marked as NaN — imputation is the
    experiment's decision, not the loader's, and a loader that quietly fills
    holes is a loader that decides how missing data behaves for everyone
    downstream.
    """
    import csv
    import io
    import zipfile

    root = data_dir() / "beijing"
    archive = fetch(BEIJING_URL, root / "beijing.zip", BEIJING_SHA)
    outer = zipfile.ZipFile(archive)
    inner_name = next(n for n in outer.namelist() if n.endswith(".zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    csv_names = sorted(n for n in inner.namelist() if n.endswith(".csv"))

    stations, columns = [], []
    for name in csv_names:
        rows = list(csv.DictReader(io.StringIO(inner.read(name).decode())))
        stations.append(rows[0]["station"])
        series = torch.full((len(rows), len(POLLUTANTS)), float("nan"))
        for t, row in enumerate(rows):
            for j, pollutant in enumerate(POLLUTANTS):
                value = row[pollutant]
                if value not in ("NA", "", None):
                    series[t, j] = float(value)
        columns.append(series)

    n = min(s.shape[0] for s in columns)
    series = torch.stack([s[:n] for s in columns], dim=1)  # (T, station, pollutant)
    return {
        "series": series,
        "names": ("station", "pollutant"),
        "stations": tuple(stations),
        "pollutants": POLLUTANTS,
    }


if __name__ == "__main__":  # `python -m examples.repro.data` pre-fetches everything
    m = mnist()
    print("mnist", {k: tuple(v.shape) for k, v in m.items()})
    c = cifar10()
    print("cifar10", {k: tuple(v.shape) for k, v in c.items()})
    b = beijing()
    print("beijing", tuple(b["series"].shape), b["stations"][:3])
