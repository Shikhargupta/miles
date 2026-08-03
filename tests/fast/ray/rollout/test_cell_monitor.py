from types import SimpleNamespace

import pytest

from miles.ray.rollout.cell_monitor import create_rollout_cell_health_checker
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig


def _make_config(**overrides) -> SimpleHealthCheckerConfig:
    defaults = dict(interval=10.0, timeout=7.0, first_wait=0.0, failure_threshold=3)
    return SimpleHealthCheckerConfig(**{**defaults, **overrides})


class _FakeApiClient:
    def __init__(self, *, exception: Exception | None = None) -> None:
        self.timeouts: list[float] = []
        self._exception = exception

    async def health_generate(self, timeout: float) -> bool:
        self.timeouts.append(timeout)
        if self._exception is not None:
            raise self._exception
        return True


def _make_cell(api_client: _FakeApiClient, *, cell_id: str = "cell-0") -> SimpleNamespace:
    return SimpleNamespace(api_client=api_client, meta=SimpleNamespace(cell_id=cell_id))


class TestCreateRolloutCellHealthChecker:
    async def test_check_probes_the_engine_of_its_own_cell(self):
        """The checker probes /health_generate of the engine the router routes to."""
        api_client = _FakeApiClient()
        checker = create_rollout_cell_health_checker(cell=_make_cell(api_client), config=_make_config())

        await checker._check_fn()

        assert len(api_client.timeouts) == 1

    async def test_check_bounds_the_probe_with_the_configured_timeout(self):
        """A hung engine must not outlive the configured health check timeout."""
        api_client = _FakeApiClient()
        checker = create_rollout_cell_health_checker(cell=_make_cell(api_client), config=_make_config(timeout=3.5))

        await checker._check_fn()

        assert api_client.timeouts == [3.5]

    async def test_check_propagates_engine_failures(self):
        """A dead engine surfaces as an exception so the checker counts it as a failure."""
        api_client = _FakeApiClient(exception=RuntimeError("engine gone"))
        checker = create_rollout_cell_health_checker(cell=_make_cell(api_client), config=_make_config())

        with pytest.raises(RuntimeError, match="engine gone"):
            await checker._check_fn()

    async def test_check_reads_the_api_client_lazily(self):
        """The engine url is only known once the cell is added, so it is resolved per check."""
        cell = _make_cell(_FakeApiClient())
        checker = create_rollout_cell_health_checker(cell=cell, config=_make_config())

        replacement = _FakeApiClient()
        cell.api_client = replacement
        await checker._check_fn()

        assert len(replacement.timeouts) == 1

    def test_name_identifies_the_cell(self):
        """Health checker logs must be attributable to a single cell."""
        checker = create_rollout_cell_health_checker(
            cell=_make_cell(_FakeApiClient(), cell_id="inference-engine-0-2"), config=_make_config()
        )

        assert checker._name == "rollout-cell-inference-engine-0-2"

    def test_the_configured_failure_threshold_is_honored(self):
        """Rollout debouncing is configurable, not pinned to a single failed probe."""
        checker = create_rollout_cell_health_checker(
            cell=_make_cell(_FakeApiClient()), config=_make_config(failure_threshold=4)
        )

        assert checker._config.failure_threshold == 4
