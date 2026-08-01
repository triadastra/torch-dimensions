"""ViT and the joint (flatten) method of multidimensionality.

The claim under test is that ViT is not a special model but a composition
choice: the same patches and the same parameter count, with a different answer
to "how are the extra axes handled".
"""

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.models.vit import PatchEmbed, ViT

# -- the flatten method -------------------------------------------------------


def _factory(**kw):
    def build(lat, d_model, plan=None):
        return td.Transformer(
            d_model,
            lat.n_axes,
            lat,
            plan=plan,
            method=td.flatten,
            mixer_kwargs={"n_heads": 2},
            **kw,
        )

    return build


def test_flatten_passes_every_applicable_check():
    report = td.testing.check_block(_factory(), d_model=4, ranks=(1, 2, 3))
    assert report, str(report)


def test_flatten_sees_every_cell_in_one_layer():
    """The defining property: one layer, and every cell can reach every other.

    The axial methods cannot do this — a single axial layer mixes along one
    axis only, so two cells differing in two coordinates are unreachable from
    each other until the second layer.
    """
    lat = td.Lattice(shape=(3, 4), names=("h", "w"))
    torch.manual_seed(0)
    joint = td.Transformer(8, 1, lat, method=td.flatten, mixer_kwargs={"n_heads": 2}).double()
    axial = td.Transformer(8, 1, lat, mixer_kwargs={"n_heads": 2}).double()

    x = torch.randn(1, 3, 4, 8, dtype=torch.float64)
    bumped = x.clone()
    # A cell differing from (0, 0) along *both* axes. The perturbation must
    # vary across features: adding a constant to every feature of a token is
    # removed exactly by the pre-norm LayerNorm, so a constant bump is
    # invisible to any pre-norm block and would make this test vacuous.
    torch.manual_seed(1)
    bumped[0, 2, 3] += torch.randn(8, dtype=torch.float64) * 10.0

    joint_reach = (joint(x) - joint(bumped))[0, 0, 0].abs().max().item()
    axial_reach = (axial(x) - axial(bumped))[0, 0, 0].abs().max().item()
    assert joint_reach > 1e-6, "joint attention did not reach across both axes in one layer"
    assert axial_reach < 1e-12, "a single axial layer must not reach across two axes"


def test_absent_cells_are_dropped_from_the_sequence_not_masked_into_it():
    """The flatten family's one structural advantage on sparse lattices."""
    valid = torch.ones(3, 4, dtype=torch.bool)
    valid[1, 1] = valid[2, 3] = False
    lat = td.Lattice(shape=(3, 4), names=("h", "w"), valid=valid)
    model = td.Transformer(8, 2, lat, method=td.flatten, mixer_kwargs={"n_heads": 2})
    assert model.nd.seq_len == 10 == lat.n_valid


def test_time_can_be_joined_or_kept_out_of_the_sequence():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"), time=True)
    joined = td.Transformer(8, 2, lat, method=td.flatten, mixer_kwargs={"n_heads": 2}).double()
    apart = td.Transformer(
        8, 2, lat, method=td.flatten, join_time=False, mixer_kwargs={"n_heads": 2}
    ).double()
    x = torch.randn(2, 5, 2, 3, 8, dtype=torch.float64)
    assert joined(x).shape == apart(x).shape == x.shape

    # With time held out, each timestep is mixed alone: perturbing t=4 cannot
    # reach t=0. With time joined, it can. Non-constant, per the note above.
    torch.manual_seed(1)
    bumped = x.clone()
    bumped[:, 4] += torch.randn(2, 3, 8, dtype=torch.float64) * 10.0
    assert (apart(x) - apart(bumped))[:, 0].abs().max().item() < 1e-12
    assert (joined(x) - joined(bumped))[:, 0].abs().max().item() > 1e-6


def test_flatten_refuses_to_be_the_model_without_a_mixer():
    lat = td.Lattice(shape=(3, 4), names=("h", "w"))
    with pytest.raises(ValueError, match="no operator at all"):
        td.flatten(None, td.ScanPlan.cyclic(lat.axis_names, 2), lat, 8)


def test_the_spec_says_no_axis_is_swept():
    lat = td.Lattice(shape=(3, 4), names=("h", "w"))
    s = td.spec(td.Transformer(8, 2, lat, method=td.flatten, mixer_kwargs={"n_heads": 2}))
    assert s["nd_method"]["family"] == "flatten"
    assert [layer["kind"] for layer in s["layers"]] == ["flatten", "flatten"]
    assert all(layer["axis"] is None for layer in s["layers"])
    assert s["sweeps"]["joint_axes"] == ["h", "w"]
    assert s["sweeps"]["directions"] == {}


# -- patch embedding ----------------------------------------------------------


@pytest.mark.parametrize(
    "image,patch,grid",
    [((32, 32), 4, (8, 8)), ((32, 32), (4, 8), (8, 4)), ((8, 8, 8), 2, (4, 4, 4))],
)
def test_patch_embed_is_rank_generic(image, patch, grid):
    embed = PatchEmbed(image, patch, in_channels=2, d_model=16)
    assert embed.grid == grid
    x = torch.randn(3, *image, 2)
    assert embed(x).shape == (3, *grid, 16)
    assert embed.lattice().shape == grid


def test_patch_embed_preserves_patch_contents():
    """Each output cell must be a function of exactly its own patch.

    A permute bug here shuffles pixels between patches and produces a model
    that trains, badly, forever — the archetypal N-D failure this library
    exists to make loud.
    """
    embed = PatchEmbed((4, 4), 2, in_channels=1, d_model=8).double()
    x = torch.zeros(1, 4, 4, 1, dtype=torch.float64)
    base = embed(x)
    x[0, 0, 0, 0] = 1.0  # a pixel in patch (0, 0) only
    out = embed(x)
    changed = (out - base).abs().sum(dim=-1)[0] > 1e-12
    assert changed[0, 0] and changed.sum() == 1, changed


def test_a_partial_patch_is_refused_rather_than_cropped():
    with pytest.raises(ValueError, match="divide the image exactly"):
        PatchEmbed((32, 30), 4, in_channels=3, d_model=8)


# -- the model ----------------------------------------------------------------


def test_vit_returns_per_patch_features():
    model = ViT(32, 2, image=(16, 16), patch=4, in_channels=3, n_heads=2)
    assert model.grid == (4, 4)
    assert model(torch.randn(2, 16, 16, 3)).shape == (2, 4, 4, 32)


def test_the_method_is_the_only_difference_between_vit_and_axial_vit():
    """The library's central claim, at its most literal: same patches, same
    parameter count, one argument apart."""
    kw = dict(image=(16, 16), patch=4, in_channels=3, n_heads=2)
    joint = ViT(32, 4, **kw)
    axial = ViT(32, 4, method=td.axial_scan, **kw)
    assert sum(p.numel() for p in joint.parameters()) == sum(p.numel() for p in axial.parameters())
    assert type(joint.nd).__name__ == "Flatten"
    assert type(axial.nd).__name__ == "AxialScan"
    x = torch.randn(2, 16, 16, 3)
    assert joint(x).shape == axial(x).shape
    # Same shape, different computation — otherwise the comparison is empty.
    assert not torch.equal(joint(x), axial(x))


def test_factorized_position_embedding_is_smaller_and_still_positional():
    kw = dict(image=(16, 16), patch=2, in_channels=1, n_heads=2)
    fac = ViT(32, 1, pos_embed="factorized", **kw)
    full = ViT(32, 1, pos_embed="full", **kw)
    n_fac = sum(p.numel() for p in fac.pos.parameters())
    n_full = sum(p.numel() for p in full.pos.parameters())
    assert n_fac == 2 * 8 * 32 and n_full == 8 * 8 * 32
    assert n_fac < n_full

    # "Positional" means two identical patches at different positions get
    # different embeddings. Without that the table is decoration.
    torch.manual_seed(0)
    for m in (fac, full):
        torch.nn.init.trunc_normal_(m.pos.tables[0], std=0.5)
        x = torch.zeros(1, 16, 16, 1)
        out = m.patch_embed(x)
        assert not torch.allclose(m.pos(out)[0, 0, 0], m.pos(out)[0, 1, 0])


def test_vit_refuses_a_lattice_it_would_have_to_reconcile():
    with pytest.raises(ValueError, match="builds its lattice"):
        ViT(32, 1, image=(8, 8), patch=2, lattice=td.Lattice(shape=(4, 4)))


def test_vit_builds_from_config_and_round_trips(tmp_path):
    model = td.build(
        {
            "kind": "vit",
            "d_model": 32,
            "n_layers": 2,
            "image": [16, 16],
            "patch": 4,
            "in_channels": 3,
            "n_heads": 2,
        }
    ).eval()
    x = torch.randn(2, 16, 16, 3)
    path = tmp_path / "vit.td"
    model.save(path)
    same = td.load(path).eval()
    assert torch.equal(model(x), same(x))
    assert same.grid == (4, 4)


def test_vit_learns():
    """A task that needs the patch grid: predict each patch's mean intensity
    from an image where the informative pixel sits in a different patch."""
    torch.manual_seed(0)
    model = ViT(32, 2, image=(8, 8), patch=2, in_channels=1, n_heads=2)
    head = torch.nn.Linear(32, 1)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=3e-3)

    def draw(g):
        x = torch.randn(16, 8, 8, 1, generator=g)
        y = x.reshape(16, 4, 2, 4, 2, 1).mean(dim=(2, 4))  # per-patch mean
        return x, y

    g = torch.Generator().manual_seed(0)
    first = last = 0.0
    for i in range(120):
        x, y = draw(g)
        loss = (head(model(x)) - y).pow(2).mean()
        first = loss.item() if i == 0 else first
        last = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert last < first / 3, (first, last)


# -- refusals and reporting ---------------------------------------------------


def test_patch_rank_must_match_the_image():
    with pytest.raises(ValueError, match="axes, image has"):
        PatchEmbed((16, 16), (4, 4, 4), in_channels=1, d_model=8)


@pytest.mark.parametrize(
    "bad,match",
    [
        (torch.randn(2, 16, 16), "expected a 4-D tensor"),
        (torch.randn(2, 8, 8, 3), "expected image dims"),
        (torch.randn(2, 16, 16, 5), "expected 3 channels"),
    ],
)
def test_malformed_input_is_refused_by_name(bad, match):
    embed = PatchEmbed((16, 16), 4, in_channels=3, d_model=8)
    with pytest.raises(ValueError, match=match):
        embed(bad)


def test_patch_embed_repr_states_the_grid_it_produces():
    text = repr(PatchEmbed((32, 32), 4, in_channels=3, d_model=8))
    assert "grid=(8, 8)" in text and "patch=(4, 4)" in text


def test_unknown_pos_embed_is_refused_and_none_is_honoured():
    with pytest.raises(ValueError, match="factorized\\|full\\|none"):
        ViT(16, 1, image=(8, 8), patch=2, in_channels=1, pos_embed="sinusoidal", n_heads=2)

    plain = ViT(16, 1, image=(8, 8), patch=2, in_channels=1, pos_embed="none", n_heads=2)
    assert sum(p.numel() for p in plain.pos.parameters()) == 0
    # With no positional embedding the stage is a pass-through, and the model
    # still runs — "none" is a supported choice, not a broken one.
    x = torch.randn(1, 8, 8, 1)
    assert torch.equal(plain.pos(plain.patch_embed(x)), plain.patch_embed(x))
    assert plain(x).shape == (1, 4, 4, 16)
    assert "none" in repr(plain.pos)


def test_pos_embed_repr_reports_its_parameter_cost():
    fac = ViT(16, 1, image=(8, 8), patch=2, in_channels=1, n_heads=2)
    assert "factorized" in repr(fac.pos) and "grid=(4, 4)" in repr(fac.pos)


# -- the flatten composition's own edges --------------------------------------


def test_flatten_reports_what_it_spans():
    lat = td.Lattice(shape=(3, 4), names=("h", "w"), time=True)
    nd = td.Transformer(8, 2, lat, method=td.flatten, mixer_kwargs={"n_heads": 2}).nd
    assert "space+time" in repr(nd) and "tokens=12" in repr(nd)
    apart = td.Transformer(
        8, 2, lat, method=td.flatten, join_time=False, mixer_kwargs={"n_heads": 2}
    ).nd
    assert "space only" in repr(apart)


def test_flatten_refuses_a_wrong_width_and_a_shape_changing_mixer():
    lat = td.Lattice(shape=(3, 4), names=("h", "w"))
    model = td.Transformer(8, 1, lat, method=td.flatten, mixer_kwargs={"n_heads": 2})
    with pytest.raises(ValueError, match="expected 8 features"):
        model(torch.randn(1, 3, 4, 5))

    class Truncating(torch.nn.Module):
        def __init__(self, d_model, **_):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            return x[:, :-1] * self.scale

    bad = td.Transformer(8, 1, lat, method=td.flatten, mixer=Truncating)
    with pytest.raises(ValueError, match="mixer changed shape"):
        bad(torch.randn(1, 3, 4, 8))


def test_flatten_chunking_matches_the_unchunked_path():
    """`chunk` exists to keep a fused kernel inside its grid limits; it must
    not change the answer."""
    lat = td.Lattice(shape=(3, 4), names=("h", "w"), time=True)
    torch.manual_seed(0)
    whole = (
        td.Transformer(8, 2, lat, method=td.flatten, mixer_kwargs={"n_heads": 2}).double().eval()
    )
    torch.manual_seed(0)
    piece = (
        td.Transformer(8, 2, lat, method=td.flatten, chunk=1, mixer_kwargs={"n_heads": 2})
        .double()
        .eval()
    )
    x = torch.randn(3, 2, 3, 4, 8, dtype=torch.float64)
    assert torch.allclose(whole(x), piece(x), rtol=0, atol=1e-12)


def test_a_shared_mixer_instance_is_used_by_every_layer():
    lat = td.Lattice(shape=(3, 4), names=("h", "w"))
    shared = td.mixers.AttentionMixer(8, 2)
    nd = td.Flatten(
        mixer=shared, plan=td.ScanPlan.cyclic(lat.axis_names, 3), lattice=lat, d_model=8
    )
    assert all(m is shared for m in nd.mixers)
    assert nd(torch.randn(1, 3, 4, 8)).shape == (1, 3, 4, 8)
