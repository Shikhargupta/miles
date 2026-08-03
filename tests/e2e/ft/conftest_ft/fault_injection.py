# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import dataclasses
import enum
import logging
import random
import threading
import time
from collections.abc import Callable

import requests

from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)

API_SERVER_PORT: int = 18080
MEAN_INTERVAL_SECONDS: float = 60.0
# Poll cell liveness this often so the gate tracks a crash->detect->heal cycle even when it
# happens entirely between two (much sparser) injections; injections still fire on the long
# random interval above.
POLL_INTERVAL_SECONDS: float = 2.0
FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]


def cell_is_alive(cell: dict) -> bool:
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


class _CellState(enum.Enum):
    INJECTED = enum.auto()  # we crashed it; the api server may still report it Healthy
    RECOVERING = enum.auto()  # observed unhealthy; awaiting its return to Healthy


class RecoveryGate:
    def __init__(self) -> None:
        self._state_of_cell_name: dict[str, _CellState] = {}

    def note_injected(self, cell_name: str) -> None:
        self._state_of_cell_name[cell_name] = _CellState.INJECTED

    def observe(self, cells_by_name: dict[str, dict]) -> None:
        for name, state in list(self._state_of_cell_name.items()):
            cell = cells_by_name.get(name)
            if cell is None or not cell_is_alive(cell):
                self._state_of_cell_name[name] = _CellState.RECOVERING
            elif state is _CellState.RECOVERING:
                del self._state_of_cell_name[name]

    def genuinely_alive(self, cells: list[dict]) -> list[dict]:
        return [c for c in cells if cell_is_alive(c) and c["metadata"]["name"] not in self._state_of_cell_name]


class ObservedCellState(enum.Enum):
    SUSPENDED = "Suspended"  # torn down, holding no gpu
    PENDING = "Pending"  # allocated but gated: no engine serving yet
    RUNNING_NOT_SERVING = "RunningNotServing"  # engine is up but not registered in the router
    SERVING = "Serving"  # registered in the router, i.e. actually able to answer requests


_RELAUNCH_STATES: tuple[ObservedCellState, ...] = (ObservedCellState.SUSPENDED, ObservedCellState.PENDING)


def compute_observed_cell_state(cell: dict) -> ObservedCellState:
    phase = cell["status"]["phase"]
    if phase == "Suspended":
        return ObservedCellState.SUSPENDED
    if phase == "Pending":
        return ObservedCellState.PENDING
    serving = any(cond["type"] == "Serving" and cond["status"] == "True" for cond in cell["status"]["conditions"])
    return ObservedCellState.SERVING if serving else ObservedCellState.RUNNING_NOT_SERVING


class _RecoveryStage(enum.Enum):
    AWAITING_RELAUNCH = enum.auto()
    AWAITING_SERVING = enum.auto()


class RecoveryWitness:
    """Pairs every accepted injection with one completed relaunch-and-serve cycle of the same cell."""

    def __init__(self) -> None:
        self.states_of_cell_name: dict[str, list[ObservedCellState]] = {}
        self._cell_type_of_name: dict[str, str] = {}
        self._num_injections_of_cell_name: dict[str, int] = {}
        self._num_recoveries_of_cell_name: dict[str, int] = {}
        self._stages_of_cell_name: dict[str, list[_RecoveryStage]] = {}

    def note_injected(self, cell_name: str) -> None:
        self._num_injections_of_cell_name[cell_name] = self._num_injections_of_cell_name.get(cell_name, 0) + 1
        self._stages_of_cell_name.setdefault(cell_name, []).append(_RecoveryStage.AWAITING_RELAUNCH)

    def observe(self, cells: list[dict]) -> None:
        for cell in cells:
            name = cell["metadata"]["name"]
            self._cell_type_of_name[name] = _cell_type_of(cell)
            state = compute_observed_cell_state(cell)
            states = self.states_of_cell_name.setdefault(name, [])
            if not states or states[-1] != state:
                states.append(state)
            self._advance(cell_name=name, state=state)

    def num_injections(self, *, cell_type: str | None = None) -> int:
        return self._total(self._num_injections_of_cell_name, cell_type=cell_type)

    def num_completed_recoveries(self, *, cell_type: str | None = None) -> int:
        return self._total(self._num_recoveries_of_cell_name, cell_type=cell_type)

    def cells_with_unfinished_recovery(self, *, cell_type: str | None = None) -> dict[str, int]:
        return {
            name: len(stages)
            for name, stages in self._stages_of_cell_name.items()
            if stages and self._matches(name, cell_type)
        }

    def _advance(self, *, cell_name: str, state: ObservedCellState) -> None:
        stages = self._stages_of_cell_name.get(cell_name)
        if not stages:
            return
        if stages[0] is _RecoveryStage.AWAITING_RELAUNCH and state in _RELAUNCH_STATES:
            stages[0] = _RecoveryStage.AWAITING_SERVING
        elif stages[0] is _RecoveryStage.AWAITING_SERVING and state is ObservedCellState.SERVING:
            stages.pop(0)
            self._num_recoveries_of_cell_name[cell_name] = self._num_recoveries_of_cell_name.get(cell_name, 0) + 1

    def _total(self, counts_of_cell_name: dict[str, int], *, cell_type: str | None) -> int:
        return sum(count for name, count in counts_of_cell_name.items() if self._matches(name, cell_type))

    def _matches(self, cell_name: str, cell_type: str | None) -> bool:
        return cell_type is None or self._cell_type_of_name.get(cell_name) == cell_type


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    stop_event: threading.Event,
    on_successful_injection: Callable[[], None],
    cell_type: str | None,
    recovery_witness: RecoveryWitness,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> None:
    rng = random.Random(seed)
    gate = RecoveryGate()
    next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_type=cell_type)
        if cells is None:
            continue

        # Track recovery on every poll so a crash->detect->heal cycle that completes between two
        # sparse injections is seen, not missed (which would exclude the cell from the live set forever).
        gate.observe({c["metadata"]["name"]: c for c in cells})
        recovery_witness.observe(cells)

        if time.monotonic() < next_injection_time:
            continue

        # Keep >=1 cell of each kind genuinely alive: if a prior injection has not recovered yet, wait
        # and retry on a later poll rather than killing that kind's last live replica.
        alive_of_type: dict[str, list[dict]] = {}
        for cell in gate.genuinely_alive(cells):
            alive_of_type.setdefault(_cell_type_of(cell), []).append(cell)
        spare_types = sorted(kind for kind, kind_cells in alive_of_type.items() if len(kind_cells) > 1)
        if not spare_types:
            logger.info(
                "Deferring injection: no cell kind has a spare replica (%s)",
                {kind: len(kind_cells) for kind, kind_cells in sorted(alive_of_type.items())},
            )
            continue

        target = rng.choice(alive_of_type[rng.choice(spare_types)])
        cell_name = target["metadata"]["name"]
        mode = rng.choice(FAILURE_MODES)
        try:
            resp = requests.post(
                f"{base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": 0},
                timeout=5,
            )
            resp.raise_for_status()
            gate.note_injected(cell_name)
            recovery_witness.note_injected(cell_name)
            on_successful_injection()
            next_injection_time = _compute_next_injection_time(rng, mean_interval_seconds)
        except Exception:
            logger.info("Failed to inject fault into %s", cell_name, exc_info=True)


def list_cells(*, base_url: str, cell_type: str | None) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if _matches_cell_type(c, cell_type)]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None


def _cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


def _matches_cell_type(cell: dict, cell_type: str | None) -> bool:
    return cell_type is None or _cell_type_of(cell) == cell_type


class FaultInjectorHandle:
    def __init__(self, *, base_url: str, seed: int, mean_interval_seconds: float, cell_type: str | None) -> None:
        self.num_successful_injections: int = 0
        self.recovery_witness = RecoveryWitness()
        self._base_url = base_url
        self._cell_type = cell_type
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=run_fault_injection_loop,
            kwargs={
                "base_url": base_url,
                "seed": seed,
                "mean_interval_seconds": mean_interval_seconds,
                "stop_event": self._stop_event,
                "on_successful_injection": self._on_successful_injection,
                "cell_type": cell_type,
                "recovery_witness": self.recovery_witness,
            },
            daemon=True,
            name="ft-random-fault-injector",
        )

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self, *, timeout_seconds: float) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout_seconds)
        self._observe_final_snapshot()

    def _observe_final_snapshot(self) -> None:
        cells = list_cells(base_url=self._base_url, cell_type=self._cell_type)
        if cells is None:
            return
        self.recovery_witness.observe(cells)

    def _on_successful_injection(self) -> None:
        self.num_successful_injections += 1


def spawn_fault_injector(*, seed: int, mean_interval_seconds: float, cell_type: str | None) -> FaultInjectorHandle:
    base_url = f"http://localhost:{API_SERVER_PORT}"
    handle = FaultInjectorHandle(
        base_url=base_url, seed=seed, mean_interval_seconds=mean_interval_seconds, cell_type=cell_type
    )
    handle.start()
    return handle
