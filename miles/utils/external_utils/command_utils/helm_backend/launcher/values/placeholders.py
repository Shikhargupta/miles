from __future__ import annotations

from miles.utils.workers import env_vars as worker_env_vars
from miles.utils.workers.worker_spec import BaseWorkerSpec

RENDERED_CELL_INDEX = 0

LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"

WORKER_INDEX_SENTINEL = 987654321
_WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"

_BASE_GPU_ID_SENTINEL = 987654322
_BASE_GPU_ID_PLACEHOLDER = f"$({worker_env_vars.BASE_GPU_ID_ENV_VAR})"


def rendered_gpu_ids(spec: BaseWorkerSpec, *, shares_its_node: bool) -> list[int]:
    gpus_per_pod = max(1, spec.scheduling.gpus_per_pod())
    if shares_its_node:
        return [_BASE_GPU_ID_SENTINEL] * gpus_per_pod
    return list(range(gpus_per_pod))


def with_base_gpu_id(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    sentinel = str(_BASE_GPU_ID_SENTINEL)
    _assert_sentinel_is_a_whole_token(argv, sentinel=sentinel, spec=spec, built_out_of="base gpu id")
    return [_BASE_GPU_ID_PLACEHOLDER if argument == sentinel else argument for argument in argv]


def with_worker_index(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    sentinel = str(WORKER_INDEX_SENTINEL)
    _assert_sentinel_is_a_whole_token(argv, sentinel=sentinel, spec=spec, built_out_of="pod index")
    return [_WORKER_INDEX_PLACEHOLDER if argument == sentinel else argument for argument in argv]


def _assert_sentinel_is_a_whole_token(
    argv: list[str], *, sentinel: str, spec: BaseWorkerSpec, built_out_of: str
) -> None:
    embedded = [argument for argument in argv if sentinel in argument and argument != sentinel]
    assert not embedded, (
        f"Spec '{spec.name}' builds {embedded} out of its {built_out_of}; the value is substituted a whole "
        f"argument at a time, so it has to reach the command unchanged"
    )
