from __future__ import annotations

from argparse import Namespace
from typing import Any

from tests.fast.ray.rollout.conftest import make_args as make_rollout_args


def make_args(**overrides: Any) -> Namespace:
    """Rollout make_args plus the trainer-side fields that spec computation reads."""
    defaults: dict[str, Any] = dict(
        train_backend="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        indep_dp=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sglang_server_concurrency=64,
        miles_router_timeout=None,
        miles_router_max_connections=None,
        miles_router_health_check_failure_threshold=3,
    )
    defaults.update(overrides)
    return make_rollout_args(**defaults)
