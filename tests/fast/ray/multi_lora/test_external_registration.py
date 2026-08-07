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


class TestControlOperations:
    def ready_backend(self, num_step=None):
        backend = make_backend()
        register_external(backend, num_step=num_step)
        backend.registry.record_weight_update(["X"])  # PENDING -> ACTIVE
        return backend

    def test_claim_requires_active_and_serialization(self):
        backend = self.ready_backend()
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", {"samples": [{}]})
        backend.enqueue_operation("X", "opt1", 2, "optim_step", {"adam_params": {"learning_rate": 3e-4}})
        # fb1 is still open: the optim_step must not be claimable.
        assert backend.claim_ready_control_operations() == []
        claimed = backend.operations.claim_data_operation("X", backend.registry.find("X").registration_id)
        backend.operations.complete(claimed["operation_id"], None)
        [op] = backend.claim_ready_control_operations()
        assert op["operation_id"] == "opt1" and op["slot"] == backend.registry.find("X").slot

    def test_unexecutable_kinds_keep_their_turn(self):
        backend = self.ready_backend()
        backend.enqueue_operation("X", "save1", 1, "save_state")
        assert backend.claim_ready_control_operations() == []
        assert backend.operations.get("save1")["state"] == "QUEUED"

    def test_success_advances_step_and_releases_dirty_pin(self):
        backend = self.ready_backend(num_step=2)
        record = backend.registry.find("X")
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        backend.commit_external_batch(["X"], [])  # pin dirty
        assert backend.registry.slot_pool.entry_of(record.tenant).pins == {"dirty-grads"}
        [op] = backend.claim_ready_control_operations()
        backend.complete_control_operations({op["operation_id"]: dict(ok=True, result={"grad_norm": 0.5})})
        assert record.step == 1
        assert backend.registry.slot_pool.entry_of(record.tenant).pins == set()
        assert backend.operations.get("opt1")["result"] == {"grad_norm": 0.5}

    def test_veto_fails_the_operation_and_clears_the_pin(self):
        backend = self.ready_backend()
        record = backend.registry.find("X")
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        backend.commit_external_batch(["X"], [])
        [op] = backend.claim_ready_control_operations()
        backend.complete_control_operations({op["operation_id"]: dict(ok=False, error="veto", category="server")})
        assert record.step == 0  # no clock advance
        assert backend.registry.slot_pool.entry_of(record.tenant).pins == set()
        view = backend.operations.get("opt1")
        assert view["state"] == "FAILED" and view["error_category"] == "server"

    def test_num_step_bound_deregisters_after_the_last_step(self):
        backend = self.ready_backend(num_step=1)
        backend.enqueue_operation("X", "opt1", 1, "optim_step")
        [op] = backend.claim_ready_control_operations()
        backend.complete_control_operations({op["operation_id"]: dict(ok=True)})
        from miles.ray.multi_lora.registry import AdapterState

        assert backend.registry.records["X"].state is AdapterState.RETIRING

    def test_commit_external_batch_completes_claimed_data_ops(self):
        backend = self.ready_backend()
        reg_id = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", {"samples": [{}]})
        backend.operations.claim_data_operation("X", reg_id)
        backend.commit_external_batch(["X"], ["fb1"])
        assert backend.operations.get("fb1")["state"] == "SUCCEEDED"

    def test_forward_backward_results_carry_row_ordered_logprobs(self):
        backend = self.ready_backend()
        reg_id = backend.registry.find("X").registration_id
        backend.enqueue_operation("X", "fb1", 1, "forward_backward", {"samples": [{}, {}]})
        backend.operations.claim_data_operation("X", reg_id)
        backend.commit_external_batch(["X"], ["fb1"], {"fb1": [[-0.1, -0.2], [-0.3]]})
        assert backend.operations.get("fb1")["result"] == {"logprobs": [[-0.1, -0.2], [-0.3]]}
