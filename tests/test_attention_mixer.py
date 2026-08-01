"""Attention as a mixer: td.Transformer.

The kernel family (test_attention.py) builds one operator per axis and
contracts them. This is the other construction — attention along whichever
single axis a layer sweeps — and the tests that matter are the ones that
distinguish it from a stack of linear layers: does it actually attend, is the
causal mask real, and does the N-D machinery treat it like any other mixer.
"""

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.mixers.attention import AttentionMixer


def test_conformance():
    def factory(lattice, d_model, plan=None):
        return td.Transformer(d_model, len(lattice.axis_names), lattice, plan=plan)

    report = td.testing.check_block(factory, d_model=4)
    assert report, str(report)


def test_it_learns_an_axial_task():
    def factory(lattice, d_model, plan=None):
        return td.Transformer(d_model, 3, lattice, plan=plan, mixer_kwargs={"n_heads": 2})

    scores = td.testing.check_trainable(factory, d_model=16, steps=200)
    assert scores["ratio"] > 3.0, scores


def test_positions_actually_attend_to_each_other():
    """A mixer that ignored the other positions would pass every shape check.
    Perturbing one position must move the others."""
    torch.manual_seed(0)
    mixer = AttentionMixer(8, n_heads=2).eval()
    x = torch.randn(2, 6, 8)
    with torch.no_grad():
        base = mixer(x)
        moved = x.clone()
        # A *different* vector, not a shifted one. The block is pre-norm, and
        # LayerNorm centers each position, so adding a constant to every
        # feature of one position is removed before attention ever sees it —
        # the first version of this test perturbed uniformly and "proved" that
        # attention does not attend.
        moved[:, 3] = torch.randn_like(moved[:, 3]) * 3
        other = mixer(moved)
    assert (base[:, 0] - other[:, 0]).abs().max() > 1e-3, "position 0 ignored position 3"


def test_the_causal_mask_is_real():
    """With causal=True, a change at position t must not reach any earlier
    position — bitwise, since attention is the only path between them."""
    torch.manual_seed(0)
    mixer = AttentionMixer(8, n_heads=2, causal=True).eval()
    x = torch.randn(2, 7, 8)
    with torch.no_grad():
        base = mixer(x)
        future = x.clone()
        future[:, 5:] += 100.0
        got = mixer(future)
    assert torch.equal(base[:, :5], got[:, :5]), "the future leaked into the past"
    assert not torch.equal(base[:, 5:], got[:, 5:]), "the change had no effect at all"


def test_non_causal_is_the_default_and_that_is_deliberate():
    """A mixer is not told which axis it sweeps, so masking 'the future' of a
    spatial axis would be meaningless. The default must be off."""
    mixer = AttentionMixer(8)
    assert mixer.causal is False
    x = torch.randn(1, 5, 8)
    with torch.no_grad():
        base = mixer(x)
        later = x.clone()
        later[:, 4] += 10.0
        assert not torch.equal(base[:, 0], mixer(later)[:, 0])


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError, match="does not divide"):
        AttentionMixer(10, n_heads=3)


def test_absent_cells_cannot_influence_a_transformer():
    """Softmax over a line that includes absent cells is exactly where a
    masking bug would hide: those positions are zero, not removed, and a naive
    implementation would still attend to them."""
    valid = torch.ones(3, 4, dtype=torch.bool)
    valid[1, 2] = valid[0, 0] = False
    lat = td.Lattice(shape=(3, 4), names=("h", "w"), valid=valid, time=True)
    model = td.Transformer(8, 4, lat).eval()
    x = torch.randn(2, 3, 3, 4, 8) * lat.mask(torch.float32)
    noisy = x.clone()
    noisy[:, :, 1, 2] = 1e3
    noisy[:, :, 0, 0] = -1e3
    with torch.no_grad():
        assert torch.equal(model(x), model(noisy))


def test_the_registry_and_checkpoints_know_it():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"), time=True)
    model = td.build(
        {
            "kind": "transformer",
            "d_model": 8,
            "n_layers": 3,
            "lattice": {"shape": [2, 3], "names": ["h", "w"], "time": True},
        }
    )
    assert type(model) is td.Transformer
    assert len(model.nd.plan) == 3
    assert lat.rank == 2


def test_spec_names_the_mixer():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"), time=True)
    spec = td.spec(td.Transformer(8, 3, lat))
    assert spec["layers"][0]["mixer"] == "AttentionMixer"
