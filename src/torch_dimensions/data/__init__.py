"""Getting real data into lattice layout.

Scoped by one distinction: **building a lattice from data is lattice
construction, which this library already owns. Running a training loop is
not.** Everything here is the former. There is no trainer, no optimizer, no
normalization policy, and no dataset downloads — and there never will be.

There is also no DataLoader. ``torch.utils.data.DataLoader`` is fine; this
module supplies the three things it needs::

    import torch_dimensions as td
    from torch.utils.data import DataLoader

    table = td.data.from_table(coords, times, values, names=("state", "sku"))
    windows = td.data.LatticeWindow(len(table), input_len=36, horizon=1)
    train, test = windows.split_at_time(table.times, "2025-01")

    ds = td.data.LatticeDataset(td.data.TensorSource(table.series, table.lattice), train)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=td.data.collate_lattice)

    model = td.LSTM(d_model=64, n_layers=6, lattice=table.lattice, d_input=table.n_features)
    for batch in dl:
        model(batch.x).pow(2).mean().backward()
"""

from torch_dimensions.data.collate import Batch, collate_lattice
from torch_dimensions.data.coords import CoordMap, from_coords
from torch_dimensions.data.memmap import MemmapSource, Normalizer, masked_stats
from torch_dimensions.data.source import LatticeDataset, LatticeSource, Sample, TensorSource
from torch_dimensions.data.table import LatticeTable, from_table
from torch_dimensions.data.window import LatticeWindow, Window

__all__ = [
    "Batch",
    "CoordMap",
    "LatticeDataset",
    "LatticeSource",
    "LatticeTable",
    "LatticeWindow",
    "MemmapSource",
    "Normalizer",
    "Sample",
    "TensorSource",
    "Window",
    "collate_lattice",
    "from_coords",
    "from_table",
    "masked_stats",
]
