from __future__ import annotations

import textwrap
from argparse import Namespace
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from miles.utils.types import Sample
from miles.utils.workers.worker_handle import BaseWorkerHandle


class FakeWorkerHandle(BaseWorkerHandle):
    """In-memory stand-in for a manager-launched engine worker handle."""

    def __init__(
        self,
        *,
        addr_and_ports: dict | None = None,
        shutdown_effect: Callable[[], Awaitable[None]] | None = None,
        init_effect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.addr_and_ports_value = dict(addr_and_ports or {})
        self.shutdown_effect = shutdown_effect
        self.init_effect = init_effect
        self.actor = MagicMock()

    async def wait_ready(self, *, timeout: float) -> None:
        pass

    async def init(self) -> None:
        self.calls.append("init")
        if self.init_effect is not None:
            await self.init_effect()

    async def get_addr_and_ports(self) -> dict:
        self.calls.append("get_addr_and_ports")
        return dict(self.addr_and_ports_value)

    async def shutdown(self) -> None:
        self.calls.append("shutdown")
        if self.shutdown_effect is not None:
            await self.shutdown_effect()


class FakeWorkerProvider:
    """Hands out pre-registered FakeWorkerHandles by worker name."""

    def __init__(self, handles: dict[str, FakeWorkerHandle] | None = None) -> None:
        self.handles: dict[str, FakeWorkerHandle] = dict(handles or {})

    async def get_handle(self, worker_name: str) -> FakeWorkerHandle:
        return self.handles[worker_name]

    async def get_url(self, worker_name: str) -> str:
        raise NotImplementedError

    async def watch_cells(self, reconcile_fn):
        raise NotImplementedError


class FakeWorkerCellControl:
    """Records start/stop/restart cell calls the way the manager would see them."""

    def __init__(self, events: list[tuple[str, dict]] | None = None) -> None:
        self.events = events if events is not None else []

    async def start_cell(self, *, cell_id: str) -> None:
        self.events.append(("start_cell", {"cell_id": cell_id}))

    async def restart_cell(self, *, cell_id: str) -> None:
        self.events.append(("restart_cell", {"cell_id": cell_id}))

    async def stop_cell(self, *, cell_id: str) -> None:
        self.events.append(("stop_cell", {"cell_id": cell_id}))


def fake_worker_handle(**kwargs) -> FakeWorkerHandle:
    return FakeWorkerHandle(**kwargs)


def make_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field touched by ``miles/ray/rollout/``.
    Adding a new field is fine; deleting one likely breaks tests."""
    defaults: dict[str, Any] = dict(
        # rollout core
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=4,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        over_sampling_batch_size=None,
        rollout_global_dataset=False,
        num_rollout=1,
        # batch / training
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        wandb_always_use_train_step=False,
        disable_rollout_trim_samples=False,
        balance_data=False,
        delay_split_train_data_by_dp=False,
        # advantage / reward
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        # placement / colocation
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
        # sglang router
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_policy=None,
        sglang_router_request_timeout_secs=600,
        sglang_dp_size=1,
        sglang_speculative_algorithm=None,
        sglang_config=None,
        sglang_model_routers=None,
        prefill_num_servers=None,
        # routers / session server
        use_miles_dashboard=False,
        use_miles_router=False,
        use_session_server=False,
        session_server_ip=None,
        session_server_port=None,
        # external rollout
        rollout_external=False,
        rollout_external_engine_addrs=None,
        # offload / fault tolerance
        offload_rollout=False,
        use_fault_tolerance=False,
        rollout_health_check_interval=10.0,
        rollout_health_check_timeout=30.0,
        # checkpoint / data source
        hf_checkpoint="/fake/model",
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        eval_function_path="miles.rollout.sglang_rollout.eval_generate_rollout",
        data_source_path="miles.data.dummy.DummyDataSource",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        # debug data
        save_debug_rollout_data=None,
        save_debug_trajectory_data=None,
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        ci_inject_rollout_data_path=None,
        ci_inject_rollout_data_start_rollout_id=None,
        ci_inject_rollout_data_min_match_ratio=0.9,
        # event checkpointing (event_logger.restore/snapshot in RolloutExecutor)
        save_debug_event_data=None,
        load=None,
        save=None,
        # CI
        ci_test=False,
        # dumper (sglang debug dumper integration)
        dumper_enable=False,
        dumper_inference=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_sample(
    *,
    group_index: int = 0,
    index: int = 0,
    response_length: int = 4,
    reward: float | dict | None = 1.0,
    status: Sample.Status = Sample.Status.COMPLETED,
    **overrides: Any,
) -> Sample:
    """Build a Sample with sensible defaults. Token list defaults to a length
    matching ``response_length`` so loss_mask/effective_response_length checks pass."""
    s = Sample(
        group_index=group_index,
        index=index,
        prompt="prompt",
        tokens=list(range(response_length)),
        response="response",
        response_length=response_length,
        label="label",
        reward=reward,
        status=status,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_samples_grouped(
    n_groups: int,
    group_size: int,
    *,
    rewards: list[float] | None = None,
    response_length: int = 4,
) -> list[Sample]:
    """Construct ``n_groups * group_size`` samples laid out group-by-group.

    If ``rewards`` is given, must have length n_groups*group_size."""
    total = n_groups * group_size
    if rewards is not None:
        assert len(rewards) == total, f"rewards must have length {total}, got {len(rewards)}"
    samples: list[Sample] = []
    for g in range(n_groups):
        for k in range(group_size):
            i = g * group_size + k
            r = rewards[i] if rewards is not None else float(k)
            samples.append(
                make_sample(
                    group_index=g,
                    index=i,
                    reward=r,
                    response_length=response_length,
                )
            )
    return samples


def make_sglang_config_yaml(
    *,
    name: str = "default",
    server_groups: list[dict] | None = None,
    update_weights: bool | None = None,
    model_path: str | None = None,
) -> str:
    """Render a small SglangConfig YAML for from_yaml() round-trip tests."""
    server_groups = server_groups or [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
    lines = ["sglang:", f"  - name: {name}"]
    if model_path is not None:
        lines.append(f"    model_path: {model_path}")
    if update_weights is not None:
        lines.append(f"    update_weights: {str(update_weights).lower()}")
    lines.append("    server_groups:")
    for g in server_groups:
        lines.append(f"      - worker_type: {g['worker_type']}")
        lines.append(f"        num_gpus: {g['num_gpus']}")
        if "num_gpus_per_engine" in g:
            lines.append(f"        num_gpus_per_engine: {g['num_gpus_per_engine']}")
    return "\n".join(lines) + "\n"


# --------------------------- ray fixtures ---------------------------


@pytest.fixture
def ray_actor_baseline(ray_local_mode):
    """Snapshot live ray actor count before / after a test; asserts no leak."""
    import ray

    def _count():
        try:
            return len([a for a in ray.util.list_named_actors() if a])
        except Exception:
            return 0

    before = _count()
    yield
    after = _count()
    assert after <= before, f"Ray actor leaked: before={before} after={after}"


@pytest.fixture(autouse=True)
def _autouse_subprocess_leak_check(monkeypatch):
    """Catch leaked router / session-server children (multiprocessing and Popen)."""
    import multiprocessing

    from miles.utils.workers import process_utils

    launched: list = []
    real_launch = process_utils.launch_bound_subprocess

    def _recording_launch(argv, *, envs):
        process = real_launch(argv, envs=envs)
        launched.append(process)
        return process

    monkeypatch.setattr(process_utils, "launch_bound_subprocess", _recording_launch)

    before = {p.pid for p in multiprocessing.active_children()}
    yield
    leaked_mp = {p.pid for p in multiprocessing.active_children()} - before
    leaked_popen = [p for p in launched if p.poll() is None]
    if leaked_mp or leaked_popen:
        # Tear down leaked children to avoid cascading test failures.
        for p in multiprocessing.active_children():
            if p.pid in leaked_mp:
                try:
                    p.terminate()
                    p.join(timeout=2)
                except Exception:
                    pass
        for p in leaked_popen:
            process_utils._terminate_process_tree(p)
        raise AssertionError(
            f"Subprocess leaked from previous test: mp={leaked_mp} popen={[p.pid for p in leaked_popen]}"
        )


def dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def make_dataclass_cells(
    *,
    num_cells: int = 2,
    num_gpus_per_engine: int = 1,
    gpu_offset: int = 0,
):
    """Build configured ``ServerCell``s without a provider (no actor lookup).
    Each cell starts unallocated."""
    from miles.ray.rollout.server_cell import ServerCell
    from miles.ray.specs.inference import _compute_nodes_per_engine

    args = make_args(num_gpus_per_node=8)
    nodes_per_engine = _compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    return [
        ServerCell(
            args=args,
            worker_type="regular",
            cell_id=f"cell-{cell_index}",
            num_nodes=nodes_per_engine,
            num_gpus_per_engine=num_gpus_per_engine,
            gpu_offset=gpu_offset + cell_index * min(num_gpus_per_engine, 8),
            cell_index=cell_index,
        )
        for cell_index in range(num_cells)
    ]
