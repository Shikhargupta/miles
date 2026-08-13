# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import abc
import dataclasses
import enum
import logging
import math
import random
import threading
import time
from collections.abc import Callable
from typing import Literal

import requests

from tests.e2e.ft.conftest_ft.pod_deletion import delete_one_pod_of_cell, list_pod_names_of_cell
from tests.e2e.ft.conftest_ft.pod_exec import sigkill_process_patterns_in_pod

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)

API_SERVER_PORT: int = 18080
# Poll cell liveness this often so the gate tracks a crash->detect->heal cycle even when it
# happens entirely between two (much sparser) injections; injections still fire on the long
# random interval above.
POLL_INTERVAL_SECONDS: float = 2.0
# The api server answers an injection once the worker has accepted the call, but on kubernetes that
# acknowledgement travels over rpc and is itself bounded. Give the request more room than the server
# is allowed to take, or a kill that worked gets recorded as a failed injection.
INJECT_REQUEST_TIMEOUT_SECONDS: float = 30.0
LIST_REQUEST_TIMEOUT_SECONDS: float = 5.0
# A fault form cannot be cancelled once it starts, and the slowest is a pod deletion: two kubectl
# calls, each bounded at a minute. Wait longer than that rather than judging a run beside a thread
# that is still crashing cells.
STOP_AND_JOIN_TIMEOUT_SECONDS: float = 180.0
FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]
RAY_ROLLOUT_ENGINE_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL]

DELETE_POD_FORM_NAME: str = "delete_pod"
EXEC_SIGKILL_FORM_NAME: str = "exec_sigkill"
ENGINE_CONTAINER_NAME: str = "engine"
SGLANG_PROCESS_PATTERN: str = "sglang::"

ACTOR_CELL_TYPE: str = "actor"
ROLLOUT_CELL_TYPE: str = "rollout"


class BaseFaultForm(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def inject(self, cell: dict, rng: random.Random) -> None: ...


class InjectFaultForm(BaseFaultForm):
    def __init__(self, *, base_url: str, failure_mode: FailureMode) -> None:
        self._base_url = base_url
        self._failure_mode = failure_mode

    @property
    def name(self) -> str:
        return f"inject_fault:{self._failure_mode.value}"

    def inject(self, cell: dict, rng: random.Random) -> None:
        resp = requests.post(
            f"{self._base_url}/api/v1/cells/{cell['metadata']['name']}/inject-fault",
            json={"mode": self._failure_mode.value, "sub_index": 0},
            timeout=INJECT_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()


class DeletePodFaultForm(BaseFaultForm):
    def __init__(self, *, namespace: str, run_id: str) -> None:
        assert namespace, "Deleting a cell's pod needs the namespace the run was installed into"
        assert run_id, "Deleting a cell's pod needs the run_id naming the release that owns it"

        self._namespace = namespace
        self._release = RunNames.release(run_id=run_id)

    @property
    def name(self) -> str:
        return DELETE_POD_FORM_NAME

    def inject(self, cell: dict, rng: random.Random) -> None:
        delete_one_pod_of_cell(
            namespace=self._namespace, release=self._release, cell_id=cell["metadata"]["name"], rng=rng
        )


class ExecSigkillFaultForm(BaseFaultForm):
    def __init__(self, *, namespace: str, run_id: str, container: str, process_pattern: str) -> None:
        assert namespace, "Crashing a process inside a cell's pod needs the namespace the run was installed into"
        assert run_id, "Crashing a process inside a cell's pod needs the run_id naming the release that owns it"

        self._namespace = namespace
        self._release = RunNames.release(run_id=run_id)
        self._container = container
        self._process_pattern = process_pattern

    @property
    def name(self) -> str:
        return EXEC_SIGKILL_FORM_NAME

    def inject(self, cell: dict, rng: random.Random) -> None:
        cell_id = cell["metadata"]["name"]
        pod_names = list_pod_names_of_cell(namespace=self._namespace, release=self._release, cell_id=cell_id)
        assert pod_names, f"Release {self._release} has no pod of cell {cell_id} in {self._namespace} to crash"

        sigkill_process_patterns_in_pod(
            namespace=self._namespace,
            pod_name=rng.choice(pod_names),
            container=self._container,
            process_pattern=self._process_pattern,
        )


CellFaultForms = dict[str, list[BaseFaultForm]]


@dataclasses.dataclass
class InjectionTally:
    num_attempts: int = 0
    num_successes: int = 0


def _assert_usable_crash_interval(mean_interval_seconds: float) -> None:
    assert math.isfinite(mean_interval_seconds) and mean_interval_seconds > 0, (
        f"A crash interval of {mean_interval_seconds}s is not a duration. Zero divides by zero inside the injector "
        f"thread, and a negative one puts every deadline in the past, so the soak crashes cells as fast as it can "
        f"poll instead of turning injection off"
    )


def create_cell_fault_forms(*, base_url: str, config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    actor_kill_forms = _inject_fault_forms(base_url=base_url, failure_modes=FAILURE_MODES)

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {
                ACTOR_CELL_TYPE: actor_kill_forms,
                ROLLOUT_CELL_TYPE: _inject_fault_forms(
                    base_url=base_url, failure_modes=RAY_ROLLOUT_ENGINE_FAILURE_MODES
                ),
            }
        case ClusterBackend.KUBERNETES:
            delete_pod_form = DeletePodFaultForm(namespace=config.namespace, run_id=config.run_id)
            exec_sigkill_form = ExecSigkillFaultForm(
                namespace=config.namespace,
                run_id=config.run_id,
                container=ENGINE_CONTAINER_NAME,
                process_pattern=SGLANG_PROCESS_PATTERN,
            )
            return {
                ACTOR_CELL_TYPE: [*actor_kill_forms, delete_pod_form],
                ROLLOUT_CELL_TYPE: [exec_sigkill_form, delete_pod_form],
            }


def _inject_fault_forms(*, base_url: str, failure_modes: list[FailureMode]) -> list[BaseFaultForm]:
    return [InjectFaultForm(base_url=base_url, failure_mode=failure_mode) for failure_mode in failure_modes]


CELL_TYPE_OF_FT_COMPONENT: dict[str, str] = {"train": ACTOR_CELL_TYPE, "rollout": ROLLOUT_CELL_TYPE}


def compute_mean_interval_seconds_of_cell_type(
    ft_components: tuple[str, ...], *, trainer_crash_interval_seconds: float, rollout_crash_interval_seconds: float
) -> dict[str, float]:
    interval_seconds_of_component: dict[str, float] = {
        "train": trainer_crash_interval_seconds,
        "rollout": rollout_crash_interval_seconds,
    }

    return {
        CELL_TYPE_OF_FT_COMPONENT[component]: interval_seconds_of_component[component] for component in ft_components
    }


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


def _cell_can_serve(cell: dict) -> bool:
    if _cell_type_of(cell) != ROLLOUT_CELL_TYPE:
        return True
    return compute_observed_cell_state(cell) is ObservedCellState.SERVING


@dataclasses.dataclass(frozen=True)
class _CellEvent:
    kind: Literal["injected", "observed"]
    state: ObservedCellState | None = None


@dataclasses.dataclass
class _CellInfo:
    cell_type: str | None = None
    events: list[_CellEvent] = dataclasses.field(default_factory=list)


class RecoveryWitness:
    """Pairs every accepted injection with one completed relaunch-and-serve cycle of the same cell."""

    def __init__(self) -> None:
        self._info_of_cell_name: dict[str, _CellInfo] = {}

    def note_injected(self, cell_name: str) -> None:
        self._info(cell_name).events.append(_CellEvent(kind="injected"))

    def observe(self, cells: list[dict]) -> None:
        for cell in cells:
            info = self._info(cell["metadata"]["name"])
            info.cell_type = _cell_type_of(cell)
            info.events.append(_CellEvent(kind="observed", state=compute_observed_cell_state(cell)))

    @property
    def states_of_cell_name(self) -> dict[str, list[ObservedCellState]]:
        return {
            name: states
            for name, info in self._info_of_cell_name.items()
            if (states := _compute_distinct_states(info.events))
        }

    def num_injections(self, *, cell_type: str | None = None) -> int:
        return len(self.injected_cell_names(cell_type=cell_type))

    def injected_cell_names(self, *, cell_type: str | None = None) -> list[str]:
        return [
            name
            for name, info in self._info_of_cell_name.items()
            if cell_type is None or info.cell_type == cell_type
            for event in info.events
            if event.kind == "injected"
        ]

    def num_completed_recoveries(self, *, cell_type: str | None = None) -> int:
        return sum(
            _compute_recovery_tally(info.events).num_completed for info in self._matching_infos(cell_type=cell_type)
        )

    def cells_with_unfinished_recovery(self, *, cell_type: str | None = None) -> dict[str, int]:
        return {
            name: tally.num_unfinished
            for name, info in self._info_of_cell_name.items()
            if (cell_type is None or info.cell_type == cell_type)
            and (tally := _compute_recovery_tally(info.events)).num_unfinished
        }

    def _info(self, cell_name: str) -> _CellInfo:
        return self._info_of_cell_name.setdefault(cell_name, _CellInfo())

    def _matching_infos(self, *, cell_type: str | None) -> list[_CellInfo]:
        return [info for info in self._info_of_cell_name.values() if cell_type is None or info.cell_type == cell_type]


@dataclasses.dataclass(frozen=True)
class _RecoveryTally:
    num_completed: int
    num_unfinished: int


class _RecoveryStage(enum.Enum):
    AWAITING_RELAUNCH = enum.auto()
    AWAITING_SERVING = enum.auto()


def _compute_recovery_tally(events: list[_CellEvent]) -> _RecoveryTally:
    num_outstanding = 0
    num_completed = 0
    stage = _RecoveryStage.AWAITING_RELAUNCH
    for event in events:
        if event.kind == "injected":
            num_outstanding += 1
            stage = _RecoveryStage.AWAITING_RELAUNCH
            continue
        if num_outstanding == 0:
            continue
        if stage is _RecoveryStage.AWAITING_RELAUNCH and event.state in _RELAUNCH_STATES:
            stage = _RecoveryStage.AWAITING_SERVING
        elif stage is _RecoveryStage.AWAITING_SERVING and event.state is ObservedCellState.SERVING:
            num_completed += num_outstanding
            num_outstanding = 0
            stage = _RecoveryStage.AWAITING_RELAUNCH
    return _RecoveryTally(num_completed=num_completed, num_unfinished=num_outstanding)


def _compute_distinct_states(events: list[_CellEvent]) -> list[ObservedCellState]:
    states: list[ObservedCellState] = []
    for event in events:
        if event.kind == "observed" and event.state is not None and (not states or states[-1] != event.state):
            states.append(event.state)
    return states


class _FormCycles:
    def __init__(self, cell_fault_forms: CellFaultForms) -> None:
        self._cell_fault_forms = cell_fault_forms
        self._remaining_of_type: dict[str, list[BaseFaultForm]] = {}

    def draw(self, cell_type: str, rng: random.Random) -> BaseFaultForm:
        remaining = self._remaining_of_type.get(cell_type) or list(self._cell_fault_forms[cell_type])
        form = remaining.pop(rng.randrange(len(remaining)))
        self._remaining_of_type[cell_type] = remaining
        return form


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    stop_event: threading.Event,
    on_injection_attempt: Callable[[str, str, bool], None],
    recovery_witness: RecoveryWitness,
    cell_fault_forms: CellFaultForms,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> None:
    rng = random.Random(seed)
    gate = RecoveryGate()
    form_cycles = _FormCycles(cell_fault_forms)
    next_injection_time_of_cell_type: dict[str, float] = {
        cell_type: _compute_next_injection_time(rng, mean_interval_seconds)
        for cell_type, mean_interval_seconds in sorted(mean_interval_seconds_of_cell_type.items())
    }

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = list_cells(base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type))
        if cells is None:
            continue

        # Track recovery on every poll so a crash->detect->heal cycle that completes between two
        # sparse injections is seen, not missed (which would exclude the cell from the live set forever).
        gate.observe({c["metadata"]["name"]: c for c in cells})
        recovery_witness.observe(cells)

        if stop_event.is_set():
            break

        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in next_injection_time_of_cell_type.items() if now >= due_at)
        if not due_types:
            continue

        live_replicas_of_type: dict[str, list[dict]] = {}
        victims_of_type: dict[str, list[dict]] = {}
        for cell in gate.genuinely_alive(cells):
            cell_type = _cell_type_of(cell)
            victims_of_type.setdefault(cell_type, []).append(cell)
            if _cell_can_serve(cell):
                live_replicas_of_type.setdefault(cell_type, []).append(cell)

        logger.info(
            "Live replicas %s, injectable victims %s",
            {
                kind: sorted(c["metadata"]["name"] for c in kind_cells)
                for kind, kind_cells in sorted(live_replicas_of_type.items())
            },
            {
                kind: sorted(c["metadata"]["name"] for c in kind_cells)
                for kind, kind_cells in sorted(victims_of_type.items())
            },
        )

        spare_types = [kind for kind in due_types if len(live_replicas_of_type.get(kind, [])) > 1]
        if not spare_types:
            logger.info(
                "Deferring injection: no due cell kind has a spare working replica (due %s, alive %s)",
                due_types,
                {kind: len(kind_cells) for kind, kind_cells in sorted(live_replicas_of_type.items())},
            )
            continue

        cell_type = rng.choice(spare_types)
        target = rng.choice(victims_of_type[cell_type])
        cell_name = target["metadata"]["name"]
        form = form_cycles.draw(cell_type, rng)
        try:
            form.inject(target, rng)
        except Exception:
            on_injection_attempt(cell_type, form.name, False)
            logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
            continue

        gate.note_injected(cell_name)
        recovery_witness.note_injected(cell_name)
        on_injection_attempt(cell_type, form.name, True)
        next_injection_time_of_cell_type[cell_type] = _compute_next_injection_time(
            rng, mean_interval_seconds_of_cell_type[cell_type]
        )
        logger.info("Injected fault %s into %s", form.name, cell_name)


def list_cells(*, base_url: str, cell_types: set[str]) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=LIST_REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if _cell_type_of(c) in cell_types]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None


def _cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


class FaultInjectorHandle:
    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        mean_interval_seconds_of_cell_type: dict[str, float],
        cell_fault_forms: CellFaultForms,
    ) -> None:
        for interval in mean_interval_seconds_of_cell_type.values():
            _assert_usable_crash_interval(interval)

        self.recovery_witness = RecoveryWitness()
        self.tally_of_form: dict[tuple[str, str], InjectionTally] = {}
        self._base_url = base_url
        self._cell_types: set[str] = set(mean_interval_seconds_of_cell_type)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=run_fault_injection_loop,
            kwargs={
                "base_url": base_url,
                "seed": seed,
                "mean_interval_seconds_of_cell_type": mean_interval_seconds_of_cell_type,
                "stop_event": self._stop_event,
                "on_injection_attempt": self._note_injection_attempt,
                "recovery_witness": self.recovery_witness,
                "cell_fault_forms": cell_fault_forms,
            },
            daemon=True,
            name="ft-random-fault-injector",
        )

    @property
    def num_successful_injections(self) -> int:
        return sum(tally.num_successes for tally in self.tally_of_form.values())

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=STOP_AND_JOIN_TIMEOUT_SECONDS)
        assert not self._thread.is_alive(), (
            f"The fault injector was still mid-injection {STOP_AND_JOIN_TIMEOUT_SECONDS}s after being asked to "
            f"stop, so it may still crash a cell nothing will heal, and reading its witness now would race it"
        )
        self._observe_final_snapshot()

    def forms_that_never_worked(self) -> list[tuple[str, str]]:
        return sorted(key for key, tally in self.tally_of_form.items() if tally.num_successes == 0)

    def _observe_final_snapshot(self) -> None:
        cells = list_cells(base_url=self._base_url, cell_types=self._cell_types)
        if cells is None:
            return
        self.recovery_witness.observe(cells)

    def _note_injection_attempt(self, cell_type: str, form_name: str, succeeded: bool) -> None:
        tally = self.tally_of_form.setdefault((cell_type, form_name), InjectionTally())
        tally.num_attempts += 1
        tally.num_successes += int(succeeded)


def spawn_fault_injector(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    cell_fault_forms: CellFaultForms,
) -> FaultInjectorHandle:
    handle = FaultInjectorHandle(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
    )
    handle.start()
    return handle
