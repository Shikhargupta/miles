"""Slot-state sidecar: stable naming and manifest gating.

The stable name must strip EXACTLY the target slot's index — stripping a
co-tenant's would let one adapter's sidecar overwrite another slot's weights
on load."""

from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.multi_lora_checkpoint import FORMAT, find_slot_state, stable_slot_param_name


class TestStableName:
    def test_strips_only_the_target_slot(self):
        name = "decoder.layers.0.self_attention.linear_qkv.adapters.3.linear_in.weight"
        assert stable_slot_param_name(name, 3) == "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"
        # A co-tenant's index must survive untouched.
        assert stable_slot_param_name(name, 2) == name

    def test_matches_exposed_export_convention(self):
        # load_adapter consumes ".adapter." keys (the expose_adapter_slot
        # export layout) — the sidecar weights section reuses that contract.
        assert ".adapter." in stable_slot_param_name("m.adapters.0.linear_out.weight", 0)

    def test_double_digit_slots(self):
        name = "m.adapters.12.linear_in.weight"
        assert stable_slot_param_name(name, 12) == "m.adapter.linear_in.weight"
        assert stable_slot_param_name(name, 1) == name


class TestManifestGating:
    def _adapter(self, tmp_path, name="a"):
        config = SimpleNamespace(save=tmp_path, rank=8, alpha=16)
        return SimpleNamespace(name=name, registration_id="r1", slot=0, step=3, version=2, config=config)

    def test_no_manifest_means_no_sidecar(self, tmp_path):
        adapter = self._adapter(tmp_path)
        (tmp_path / "slot_state").mkdir()
        assert find_slot_state(adapter) is None

    def test_manifest_must_match_name_and_world(self, tmp_path):
        adapter = self._adapter(tmp_path)
        base = tmp_path / "slot_state"
        base.mkdir()
        torch.save(
            {"format": FORMAT, "name": "someone-else", "optimizer_step": 3, "world_size": 1},
            base / "manifest.pt",
        )
        assert find_slot_state(adapter) is None

        torch.save(
            {"format": FORMAT, "name": "a", "optimizer_step": 3, "world_size": 1},
            base / "manifest.pt",
        )
        assert find_slot_state(adapter) == base

    def test_no_save_dir_means_no_sidecar(self):
        adapter = SimpleNamespace(config=SimpleNamespace(save=None))
        assert find_slot_state(adapter) is None


class TestSwapInSidecarSentinel:
    """A sidecar that exists with optimizer step 0 (an adapter swapped out
    before its first step) must be restored as-is — only a MISSING sidecar
    (None) may fall back to the weights-only registration re-init."""

    def _swap_in_with(self, monkeypatch, load_result):
        import miles.backends.megatron_utils.multi_lora_checkpoint as mlc
        import miles.backends.megatron_utils.multi_lora_optimizer as mlo
        import miles.backends.megatron_utils.multi_lora_scheduler as mls
        import miles.backends.megatron_utils.multi_lora_utils as mlu

        calls = []
        monkeypatch.setattr(mlc, "load_slot_state", lambda *a, **k: load_result)
        monkeypatch.setattr(mlu, "_register_adapter", lambda *a, **k: calls.append("register") or 0)
        monkeypatch.setattr(mlo, "reload_adapter_slot_model_params", lambda *a, **k: calls.append("reload"))
        monkeypatch.setattr(mls, "install_slot_scheduler", lambda *a, **k: calls.append("scheduler"))
        adapter = SimpleNamespace(name="a", slot=0)
        step = mlc.swap_in(args=SimpleNamespace(), model=[], optimizer=None, adapter=adapter)
        return step, calls

    def test_step_zero_sidecar_is_not_reinitialized(self, monkeypatch):
        step, calls = self._swap_in_with(monkeypatch, load_result=0)
        assert step == 0
        assert calls == ["scheduler"]  # no register, no master re-derivation

    def test_missing_sidecar_falls_back_to_registration(self, monkeypatch):
        step, calls = self._swap_in_with(monkeypatch, load_result=None)
        assert step == 0
        assert calls == ["register", "reload", "scheduler"]
