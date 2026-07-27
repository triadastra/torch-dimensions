"""End-to-end: long-format rows -> lattice -> N-D model -> a trained model.

    python examples/train_nd.py

torch-dimensions ships no trainer, on purpose — optimizers, schedules and loops
belong to you. What it does guarantee is that its models are ordinary
``nn.Module``s that train with ordinary PyTorch. This file is that guarantee
made runnable: every line below the model construction is plain torch, and
none of it is special.

The task is one-step-ahead forecasting over a (region x product) lattice where
one combination is never sold, so the lattice is genuinely sparse.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch_dimensions as td

REGIONS = ("north", "south", "east")
PRODUCTS = ("widget", "gadget")
NEVER_SOLD = ("east", "gadget")


def synthetic_rows(n_months=120, seed=0):
    """Long-format rows, the shape a database actually hands you."""
    g = torch.Generator().manual_seed(seed)
    coords, times, values = [], [], []
    phase = {c: torch.rand(1, generator=g).item() * 6.28 for c in REGIONS + PRODUCTS}
    for t in range(n_months):
        for region in REGIONS:
            for product in PRODUCTS:
                if (region, product) == NEVER_SOLD:
                    continue  # this combination does not exist
                season = torch.sin(torch.tensor(t / 6.0 + phase[region])).item()
                trend = t / n_months
                noise = torch.randn(1, generator=g).item() * 0.05
                coords.append((region, product))
                times.append(t)
                values.append([season + trend + noise])
    return coords, times, values


def main() -> None:
    torch.manual_seed(0)

    # 1. Rows to a lattice. The grid, its axis vocabularies, and which cells
    #    exist are all inferred -- no hand-written coordinate mapping.
    table = td.data.from_table(*synthetic_rows(), names=("region", "product"))
    print(table)
    print("lattice:", table.lattice)

    # 2. Window the time axis, and split so no training window contains a
    #    timestep from after the cut.
    windows = td.data.LatticeWindow(len(table), input_len=12, horizon=1)
    train_w, test_w = windows.split(96)
    print(f"windows: {len(train_w)} train, {len(test_w)} test")

    source = td.data.TensorSource(table.series, table.lattice)
    loader = DataLoader(
        td.data.LatticeDataset(source, train_w),
        batch_size=16,
        shuffle=True,
        collate_fn=td.data.collate_lattice,
    )

    # 3. An N-D model. Time stays causal; the categorical axes are swept both
    #    ways, since nothing about "region" is directional.
    model = td.LSTM(
        d_model=32,
        n_layers=6,
        lattice=table.lattice,
        d_input=table.n_features,
        bidirectional=("region", "product"),
    )
    head = nn.Linear(32, table.n_features)
    print("plan:", model.plan)
    print(
        "params:", td.spec(model)["model"]["n_params"] + sum(p.numel() for p in head.parameters())
    )

    # 4. Your training loop. Nothing here comes from torch-dimensions.
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=3e-3)
    mask = table.lattice.mask(torch.float32)

    for epoch in range(15):
        total, n = 0.0, 0
        for batch in loader:
            pred = head(model(batch.x))[:, -1:]  # last step of the window
            loss = ((pred - batch.y) * mask).pow(2).sum() / (mask.sum() * pred.shape[0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        if epoch % 3 == 0 or epoch == 14:
            print(f"epoch {epoch:2d}  train mse {total / n:.5f}")

    # 5. Held-out evaluation on windows entirely after the split.
    model.eval()
    test_loader = DataLoader(
        td.data.LatticeDataset(source, test_w),
        batch_size=16,
        collate_fn=td.data.collate_lattice,
    )
    with torch.no_grad():
        errs = [
            ((head(model(b.x))[:, -1:] - b.y) * mask).pow(2).sum().item()
            / (mask.sum().item() * b.x.shape[0])
            for b in test_loader
        ]
    print(f"held-out mse {sum(errs) / len(errs):.5f}")

    # Absent cells stay exactly zero all the way through.
    with torch.no_grad():
        out = head(model(next(iter(test_loader)).x)) * mask
    absent = out.masked_select(~table.lattice.mask().expand_as(out))
    print(f"absent-cell outputs: max |x| = {absent.abs().max().item():.1e}")


if __name__ == "__main__":
    main()
