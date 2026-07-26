"""``RolloutExecutor`` generate/eval flow driven through the production class.

We instantiate ``RolloutExecutor.__ray_actor_class__`` directly (the raw Python
class behind ``@ray.remote``) so ``monkeypatch`` reaches its dependencies, while
``ray.put`` in the DP-split path still runs against a real Ray runtime.

The executor owns no engines: everything inference-side is reached through the
``InferenceController`` handle it is constructed with, which these tests replace
with a recording stub. Engine-side behaviour is covered by
``test_inference_controller.py``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainInput, RolloutFnTrainOutput


class _RecordingRemoteCall:
    """Stands in for ``actor.some_method`` on a Ray actor handle."""

    def __init__(self, name: str, log: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._name = name
        self._log = log

    def remote(self, *args: Any) -> asyncio.Future:
        self._log.append((self._name, args))
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        return future


class _StubInferenceController:
    """Records the lifecycle hooks the executor is expected to fire."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.prepare_rollout = _RecordingRemoteCall("prepare_rollout", self.calls)
        self.prepare_eval = _RecordingRemoteCall("prepare_eval", self.calls)


@pytest.fixture
def patch_low_level(monkeypatch):
    """No-op the executor dependencies that touch wandb / tensorboard / disk /
    not-importable default function paths."""
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
class TestGenerate:
    """``generate(rollout_id)`` is the trainer's per-iteration rollout entry
    point. It must (1) tell the inference controller a rollout is starting,
    (2) call the rollout function with ``RolloutFnTrainInput(rollout_id=N)``,
    (3) postprocess + convert + DP-split the returned samples."""

    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        args = _make_test_args()
        # global_batch_size = number of samples we'll produce (postprocess
        # trims to a multiple, so equality avoids losing samples).
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
        # generate returns {"sample_indices": ..., "data_ref": ...};
        # split_train_data_by_dp returns Box(ObjectRef) per dp rank
        assert set(result) == {"sample_indices", "data_ref"}
        data_refs = result["data_ref"]
        assert len(data_refs) == 2
        partitions = ray.get([box.inner for box in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            # 8 samples / 2 dp = 4 per rank
            assert len(partition["tokens"]) == 4

    async def test_notifies_inference_controller_before_generating(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        """The executor no longer owns health monitors or engines, so it must
        hand the rollout id to the controller — that call is what resumes health
        monitoring, runs CI fault injection and refreshes the dashboard
        topology, and it must happen before any generation work."""
        args = _make_test_args()
        args.global_batch_size = 4

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)
        executor.set_train_parallel_config({"dp_size": 1})

        call_order: list[str] = []

        def fake_rollout_fn(input):
            call_order.append("rollout_fn")
            return RolloutFnTrainOutput(samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={})

        executor.generate_rollout = fake_rollout_fn

        await executor.generate(rollout_id=7)

        assert controller.calls == [("prepare_rollout", (7,))]
        assert call_order == ["rollout_fn"]


@pytest.mark.asyncio
class TestEval:
    async def test_invokes_eval_fn_with_eval_input(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        args = _make_test_args()

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)

        captured: list = []

        def fake_eval_fn(input):
            captured.append(input)
            return RolloutFnEvalOutput(
                data={"my_dataset": {"rewards": [0.5, 1.0]}},
                metrics={},
            )

        executor.eval_generate_rollout = fake_eval_fn

        await executor.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10
        assert controller.calls == [("prepare_eval", ())]

    async def test_skipped_in_debug_train_only_mode(
        self,
        ray_local_mode,
        patch_low_level,
    ):
        """``debug_train_only=True`` must short-circuit ``eval`` before the
        rollout function is invoked — used by trainer-only debug runs that
        have no rollout cluster. The controller must not be poked either."""
        args = _make_test_args()
        args.debug_train_only = True

        controller = _StubInferenceController()
        executor = _make_executor(args, controller)

        called: list = []
        executor.eval_generate_rollout = lambda inp: called.append(inp)

        await executor.eval(rollout_id=10)

        assert called == []
        assert controller.calls == []
