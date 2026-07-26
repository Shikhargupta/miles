from __future__ import annotations

import asyncio
from typing import Any

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainInput, RolloutFnTrainOutput


class _RecordingRemoteCall:
    def __init__(self, name: str, log: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._name = name
        self._log = log

    def remote(self, *args: Any) -> asyncio.Future:
        self._log.append((self._name, args))
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        return future


class _StubInferenceController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.prepare_rollout = _RecordingRemoteCall("prepare_rollout", self.calls)
        self.prepare_eval = _RecordingRemoteCall("prepare_eval", self.calls)


@pytest.fixture
def process_setup_calls(monkeypatch) -> list[str]:
    import miles.ray.rollout.rollout_executor as rexec

    recorded: list[str] = []
    monkeypatch.setattr(rexec, "init_http_client", lambda args: recorded.append("init_http_client"))
    monkeypatch.setattr(rexec, "start_session_server", lambda args: recorded.append("start_session_server"))
    return recorded


@pytest.fixture
def patch_low_level(monkeypatch, process_setup_calls):
    import miles.ray.rollout.rollout_executor as rexec

    monkeypatch.setattr(rexec, "configure_logger", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "save_debug_rollout_data", lambda *a, **kw: None)


def _make_executor(args, inference_controller):
    return RolloutExecutor.__ray_actor_class__(args, inference_controller)


def _make_test_args(**overrides):
    return make_args(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        **overrides,
    )


@pytest.mark.asyncio
class TestProcessSetup:
    async def test_initializes_http_client_and_session_servers(
        self,
        ray_local_mode,
        patch_low_level,
        process_setup_calls,
    ):
        """The rollout functions run in this process, so their HTTP client and session servers start here."""
        args = _make_test_args()

        _make_executor(args, _StubInferenceController())

        assert process_setup_calls == ["init_http_client", "start_session_server"]

    async def test_skips_process_setup_in_debug_train_only(
        self,
        ray_local_mode,
        patch_low_level,
        process_setup_calls,
    ):
        """No engines exist in this mode, so there is nothing to talk to."""
        args = _make_test_args()
        args.debug_train_only = True

        _make_executor(args, _StubInferenceController())

        assert process_setup_calls == []


@pytest.mark.asyncio
class TestGenerate:
    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        """generate passes a train input and returns the samples split per dp rank."""
        args = _make_test_args()
        args.global_batch_size = 8

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)
        executor.set_train_parallel_config({"dp_size": 2})

        captured: list = []

        def fake_rollout_fn(input):
            captured.append(input)
            return RolloutFnTrainOutput(
                samples=[make_samples_grouped(n_groups=2, group_size=4)],
                metrics={"my_metric": 1.23},
            )

        executor.generate_rollout = fake_rollout_fn

        result = await executor.generate(rollout_id=42)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnTrainInput)
        assert captured[0].rollout_id == 42
        assert set(result) == {"sample_indices", "data_ref"}
        data_refs = result["data_ref"]
        assert len(data_refs) == 2
        partitions = ray.get([box.inner for box in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            assert len(partition["tokens"]) == 4

    async def test_notifies_inference_controller_before_generating(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        """prepare_rollout must reach the controller before any generation work starts."""
        args = _make_test_args()
        args.global_batch_size = 4

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)
        executor.set_train_parallel_config({"dp_size": 1})

        def fake_rollout_fn(input):
            controller.calls.append(("rollout_fn", (input.rollout_id,)))
            return RolloutFnTrainOutput(samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={})

        executor.generate_rollout = fake_rollout_fn

        await executor.generate(rollout_id=7)

        assert controller.calls == [("prepare_rollout", (7,)), ("rollout_fn", (7,))]


@pytest.mark.asyncio
class TestEval:
    async def test_invokes_eval_fn_with_eval_input(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        """eval passes an eval input and resumes health monitoring before issuing requests."""
        args = _make_test_args()

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)

        captured: list = []

        def fake_eval_fn(input):
            captured.append(input)
            controller.calls.append(("eval_fn", ()))
            return RolloutFnEvalOutput(
                data={"my_dataset": {"rewards": [0.5, 1.0]}},
                metrics={},
            )

        executor.eval_generate_rollout = fake_eval_fn

        await executor.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10
        assert controller.calls == [("prepare_eval", ()), ("eval_fn", ())]

    async def test_skipped_in_debug_train_only_mode(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        """debug_train_only short-circuits eval without touching the rollout fn or the controller."""
        args = _make_test_args()
        args.debug_train_only = True

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)

        called: list = []
        executor.eval_generate_rollout = lambda inp: called.append(inp)

        await executor.eval(rollout_id=10)

        assert called == []
        assert controller.calls == []
