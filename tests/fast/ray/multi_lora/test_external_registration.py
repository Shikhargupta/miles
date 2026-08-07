"""External (client-driven) adapter registration and operation routing through
MultiLoRABackend (no Ray, no HTTP I/O)."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.utils.adapter_config import AdapterRunConfig

DATA_FILE = __file__


def make_backend(max_adapters: int = 4) -> MultiLoRABackend:
    args = SimpleNamespace(
        multi_lora_n_adapters=max_adapters,
        save="/tmp/miles-test-save",
        lora_rank=32,
        lora_alpha=32,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        multi_lora_max_adapter_global_batch_size=256,
    )
    return MultiLoRABackend(args, "http://unused")


def external_config(**overrides) -> AdapterRunConfig:
    return AdapterRunConfig(input_mode="external", **overrides)


def register_external(backend, name="X", **overrides) -> dict:
    return asyncio.run(backend.register(name, external_config(**overrides)))


class TestExternalRegistration:
    def test_external_registers_without_data_or_reward(self):
        backend = make_backend()
        result = register_external(backend, rank=8, alpha=16)
        assert result == {"name": "X", "slot": 0}
        config = backend.registry.find("X").config
        assert config.rank == 8 and config.alpha == 16
        assert str(config.save).endswith("adapters/X")

    def test_dataset_mode_still_requires_data(self):
        backend = make_backend()
        with pytest.raises(ValueError, match="needs a dataset path"):
            asyncio.run(backend.register("D", AdapterRunConfig()))

    def test_unknown_input_mode_is_rejected(self):
        backend = make_backend()
        with pytest.raises(ValueError, match="input_mode"):
            asyncio.run(backend.register("D", AdapterRunConfig(input_mode="stream")))

    @pytest.mark.parametrize(
        "overrides, message",
        [
            (dict(data=DATA_FILE), "must not set 'data'"),
            (dict(rm_type="math"), "must not set a reward"),
            (dict(custom_rm_path="pkg:fn"), "must not set a reward"),
            (dict(rollout_function_path="pkg:fn"), "must not set rollout_function_path"),
            (dict(num_epoch=2), "must not set num_epoch"),
            (dict(num_step=0), "num_step must be a positive integer"),
            (dict(rank=64), "exceeds the allocated maximum rank"),
        ],
    )
    def test_external_config_rejections(self, overrides, message):
        backend = make_backend()
        with pytest.raises(ValueError, match=message):
            register_external(backend, **overrides)


class TestOperationRouting:
    def test_enqueue_resolves_the_current_registration(self):
        backend = make_backend()
        register_external(backend)
        record = backend.registry.find("X")
        view = backend.enqueue_operation("X", "op1", 1, "forward_backward", {"samples": []})
        assert view["registration_id"] == record.registration_id
        assert view["state"] == "QUEUED"

    def test_dataset_adapters_take_no_operations(self):
        backend = make_backend()
        asyncio.run(backend.register("D", AdapterRunConfig(data=DATA_FILE, rm_type="math")))
        with pytest.raises(ValueError, match="input_mode: external"):
            backend.enqueue_operation("D", "op1", 1, "forward_backward")

    def test_unregistered_name_is_rejected(self):
        backend = make_backend()
        with pytest.raises(ValueError, match="not accepting operations"):
            backend.enqueue_operation("ghost", "op1", 1, "forward_backward")

    def test_retirement_fences_open_operations(self, monkeypatch):
        backend = make_backend()
        register_external(backend)
        backend.enqueue_operation("X", "op1", 1, "forward_backward")

        async def no_abort(name, registration_id):
            pass

        monkeypatch.setattr(backend, "abort_adapter_requests", no_abort)
        asyncio.run(backend.deregister("X"))
        asyncio.run(backend.retire_adapters())
        view = backend.operations.get("op1")
        assert view["state"] == "FAILED" and view["error_category"] == "user"
        with pytest.raises(ValueError, match="not accepting operations"):
            backend.enqueue_operation("X", "op2", 2, "forward_backward")
