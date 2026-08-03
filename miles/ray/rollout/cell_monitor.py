from typing import TYPE_CHECKING

from miles.utils.ft_utils.health_checker import SimpleHealthChecker, SimpleHealthCheckerConfig

if TYPE_CHECKING:
    from miles.ray.rollout.server_cell import ServerCell


def create_rollout_cell_health_checker(
    *,
    cell: "ServerCell",
    config: SimpleHealthCheckerConfig,
) -> SimpleHealthChecker:
    async def _check() -> None:
        # Cell health is liveness of the engine that the router routes to: /health_generate
        # answers even while the engine holds only stale weights, and raises once the
        # underlying process is gone.
        await cell.api_client.health_generate(timeout=config.timeout)

    return SimpleHealthChecker(
        name=f"rollout-cell-{cell.meta.cell_id}",
        check_fn=_check,
        config=config,
    )
