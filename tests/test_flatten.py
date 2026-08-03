def test_joint_attention_is_permutation_invariant_without_positions():
    """`td.flatten` adds no positional term, so with a set-function mixer the
    whole model is a set function: permuting an axis and un-permuting the
    output changes nothing. This is why a joint-attention model cannot learn
    a cumulative sum along an axis, and why `td.ViT` carries its own
    positional embedding. Documented in `Flatten`'s docstring; pinned here so
    the claim cannot quietly stop being true."""
    import torch

    import torch_dimensions as td

    torch.manual_seed(0)
    dense = td.Lattice(shape=(6, 8), names=("h", "w"), time=True)
    model = td.Transformer(32, 2, dense, d_input=1, method=td.flatten).eval()

    x = torch.randn(1, 3, 6, 8, 1)
    perm = torch.randperm(8)
    with torch.no_grad():
        straight = model(x)
        shuffled = model(x[:, :, :, perm])[:, :, :, torch.argsort(perm)]
    assert torch.allclose(straight, shuffled, atol=1e-5)
