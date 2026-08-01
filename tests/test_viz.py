"""Track B V2 acceptance: the viewer ships in the wheel and serves a model.

The boundary under test is the same one the viewer has always had — a
versioned JSON document — so these tests speak HTTP and JSON and never import
anything JS-adjacent.
"""

import json
import urllib.error
import urllib.request

import pytest

import torch_dimensions as td
from torch_dimensions import viz

pytestmark = pytest.mark.skipif(
    not viz.bundle_exists(),
    reason="viewer bundle not built (see viewer/install_bundle.py)",
)


@pytest.fixture
def model():
    return td.S4DND(16, 4, dim=2, shape=(3, 4), time=True)


@pytest.fixture
def served(model):
    server = viz.serve(model)
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 — a localhost test server
        return r.status, r.read()


def test_the_served_spec_is_the_models_own_spec(served, model):
    status, body = get(f"{served}/spec.json")
    assert status == 200
    assert json.loads(body) == td.spec(model)


def test_the_bundle_is_served_as_a_page(served):
    status, body = get(f"{served}/")
    assert status == 200
    assert b'id="root"' in body, "index.html is not the viewer's"


def test_no_local_training_run_rides_along_in_the_bundle(served):
    """A stale `viewer/public/run.json` was baked into the first build of this
    bundle, so a wheel would have carried whatever run was on the packager's
    laptop — and the viewer loads a run in preference to the model it was
    handed, which is how it was noticed. The installer strips it; this fails
    if that ever stops happening."""
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{served}/run.json")
    assert e.value.code == 404


def test_show_accepts_a_spec_dict_and_a_json_file(tmp_path, model):
    spec = td.spec(model)
    assert viz.resolve_spec(spec) == spec
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert viz.resolve_spec(path) == spec
    assert viz.resolve_spec(str(path)) == spec


def test_show_accepts_a_checkpoint(tmp_path, model):
    ckpt = tmp_path / "model.td"
    model.save(ckpt)
    assert viz.resolve_spec(ckpt)["model"]["kind"] == "S4DND"


def test_what_cannot_be_shown_says_so_instead_of_serving_a_blank_page(tmp_path):
    with pytest.raises(TypeError, match="cannot show a int"):
        viz.resolve_spec(7)
    with pytest.raises(ValueError, match="not a spec document"):
        viz.resolve_spec({"hello": "world"})
    with pytest.raises(FileNotFoundError):
        viz.resolve_spec(tmp_path / "nope.json")


def test_a_missing_bundle_explains_how_to_build_it(monkeypatch, model):
    """A git checkout has no bundle — that is expected, and the error has to
    say what to run rather than 'index.html not found'."""
    monkeypatch.setattr(viz, "BUNDLE", viz.BUNDLE.parent / "not_built")
    with pytest.raises(FileNotFoundError, match="npm run build"):
        viz.serve(model)
