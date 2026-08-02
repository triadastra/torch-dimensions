"""The upstream-by-default flip: `portable=True` selects the light build.

The default implementation of td.S4 / td.S4D / td.Mamba is the original
authors' code (vendored, byte-verified); `portable=True` selects our
pure-torch mixers. The flag is plain data in the model's config, so
checkpoints rebuild what was trained — including checkpoints from before the
flag existed, which rebuild portable because that is what they were.
"""

from __future__ import annotations

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.mixers.ssm import MambaMixer, S4DMixer, S4Mixer
from torch_dimensions.mixers.upstream import (
    UpstreamMambaMixer,
    UpstreamS4DMixer,
    UpstreamS4Mixer,
)

pytest.importorskip("einops", reason="the default (upstream) mixers need the [upstream] extra")
pytest.importorskip("hydra", reason="the s4 pipeline needs hydra-core")

LAT = td.Lattice(shape=(3, 4), names=("y", "x"))

CASES = [
    (td.S4, S4Mixer, UpstreamS4Mixer, {}),
    (td.S4D, S4DMixer, UpstreamS4DMixer, {}),
    (td.Mamba, MambaMixer, UpstreamMambaMixer, {"d_state": 8}),
]


def _has(model, cls) -> bool:
    return any(isinstance(m, cls) for m in model.modules())


@pytest.mark.parametrize("host,portable_cls,upstream_cls,kw", CASES)
def test_default_is_upstream_and_flag_selects_portable(host, portable_cls, upstream_cls, kw):
    default = host(8, 2, lattice=LAT, **kw)
    light = host(8, 2, lattice=LAT, portable=True, **kw)
    assert _has(default, upstream_cls) and not _has(default, portable_cls)
    assert _has(light, portable_cls) and not _has(light, upstream_cls)
    assert default.config["portable"] is False
    assert light.config["portable"] is True


@pytest.mark.parametrize("portable", [False, True])
def test_save_load_round_trips_the_flag(tmp_path, portable):
    torch.manual_seed(0)
    m = td.S4D(8, 2, lattice=LAT, portable=portable).eval()
    p = tmp_path / "m.td"
    m.save(p)
    r = td.load(p).eval()
    x = torch.randn(2, 3, 4, 8)
    with torch.no_grad():
        assert torch.equal(m(x), r(x))
    assert r.config["portable"] is portable


def test_pre_flag_checkpoint_rebuilds_portable(tmp_path):
    """A checkpoint written before the flag existed records no `portable` key
    — and was built with the portable mixers. Rebuilding it must produce that
    model, bitwise, not today's default."""
    torch.manual_seed(0)
    m = td.S4D(8, 2, lattice=LAT, portable=True).eval()
    p = tmp_path / "legacy.td"
    m.save(p)
    ck = torch.load(p, weights_only=False)
    del ck["config"]["portable"]  # simulate the old format
    torch.save(ck, p)

    r = td.load(p).eval()
    assert _has(r, S4DMixer)
    x = torch.randn(2, 3, 4, 8)
    with torch.no_grad():
        assert torch.equal(m(x), r(x))


def test_portable_plus_mixer_is_refused():
    with pytest.raises(ValueError, match="not both"):
        td.S4D(8, 2, lattice=LAT, portable=True, mixer=UpstreamS4DMixer)


def test_fresh_config_gets_todays_default():
    """The legacy shim applies to checkpoint files only: a dict written today
    means today's default (upstream)."""
    m = td.build({"kind": "s4d", "d_model": 8, "n_layers": 2})
    assert _has(m, UpstreamS4DMixer)


# --- auto-install behaviour ---------------------------------------------------


def test_auto_install_disabled_gives_instructions(monkeypatch):
    import builtins

    from torch_dimensions.mixers import upstream

    real_import = builtins.__import__

    def no_einops(name, *a, **k):
        if name == "einops":
            raise ImportError("No module named 'einops'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_einops)
    monkeypatch.setenv("TD_NO_AUTO_INSTALL", "1")
    with pytest.raises(ImportError, match=r"torch-dimensions\[upstream\]"):
        upstream._require_upstream(("einops",))


def test_auto_install_invokes_pip_with_mapped_names(monkeypatch):
    """`hydra` the module maps to `hydra-core` the package; the pip command
    must carry the pip name. The fake install 'succeeds' by unhiding the
    module, so the post-install recheck passes."""
    import builtins
    import subprocess as sp

    from torch_dimensions.mixers import upstream

    hidden = {"hydra"}
    real_import = builtins.__import__

    def hider(name, *a, **k):
        if name in hidden:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        hidden.clear()  # "installation" makes it importable again

    monkeypatch.setattr(builtins, "__import__", hider)
    monkeypatch.setattr(sp, "run", fake_run)
    monkeypatch.delenv("TD_NO_AUTO_INSTALL", raising=False)
    upstream._require_upstream(("hydra",))
    assert len(calls) == 1
    assert "hydra-core" in calls[0] and "hydra" not in calls[0][4:]
