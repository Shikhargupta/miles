from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.torchtitan_utils import model as titan_model


class _FakeModel:
    def eval(self):
        self.evaluated = True

    def requires_grad_(self, flag):
        self.requires_grad = flag


def _config(offload=False):
    return SimpleNamespace(training=SimpleNamespace(enable_cpu_offload=offload))


def test_ref_model_requires_a_checkpoint_to_load():
    with pytest.raises(ValueError, match="--ref-load"):
        titan_model.build_ref_model(Namespace(ref_load=None), None, _config(), None, None)


def test_ref_model_is_built_offloaded_then_frozen(monkeypatch):
    seen = {}

    def fake_build_model(args, spec, config, parallel_dims, device):
        seen["offload_during_build"] = config.training.enable_cpu_offload
        return object(), _FakeModel()

    monkeypatch.setattr(titan_model, "build_model", fake_build_model)
    monkeypatch.setattr(titan_model, "load_hf_weights", lambda *a, **k: None)

    config = _config()
    ref = titan_model.build_ref_model(Namespace(ref_load="/ckpt"), None, config, None, None)

    assert seen["offload_during_build"] is True
    assert ref.evaluated is True
    assert ref.requires_grad is False


def test_the_offload_flag_is_restored_even_when_the_build_fails(monkeypatch):
    """The flag lives on the config the actor keeps, so leaving it set would make
    every later reader believe the actor model is offloaded too."""

    def exploding_build_model(*a, **k):
        raise RuntimeError("build blew up")

    monkeypatch.setattr(titan_model, "build_model", exploding_build_model)

    config = _config(offload=False)
    with pytest.raises(RuntimeError, match="build blew up"):
        titan_model.build_ref_model(Namespace(ref_load="/ckpt"), None, config, None, None)

    assert config.training.enable_cpu_offload is False


def test_an_already_offloaded_config_stays_offloaded(monkeypatch):
    monkeypatch.setattr(titan_model, "build_model", lambda *a, **k: (object(), _FakeModel()))
    monkeypatch.setattr(titan_model, "load_hf_weights", lambda *a, **k: None)

    config = _config(offload=True)
    titan_model.build_ref_model(Namespace(ref_load="/ckpt"), None, config, None, None)

    assert config.training.enable_cpu_offload is True
