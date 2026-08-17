import dataclasses
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import fault_injection as fi
from tests.e2e.ft.conftest_ft.modes import MODES, FTTestMode
from tests.e2e.ft.conftest_ft.scenario_random_crash import _assert_every_drawn_fault_form_worked, assert_healing

from miles.utils.external_utils import command_utils
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.workers.types import ClusterBackend

_ROLLOUT_CELL_NAME = "rollout-engine-0"


def _mode(*ft_components: str) -> FTTestMode:
    return dataclasses.replace(next(iter(MODES.values())), ft_components=tuple(ft_components))


def _injector(*, cell_type: str, num_successful_injections: int) -> fi.FaultInjectorHandle:
    forms = fi.create_cell_fault_forms(
        base_url="http://control",
        config=command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, namespace="", run_id="r"),
    )
    injector = fi.FaultInjectorHandle(
        base_url="http://control", seed=0, mean_interval_seconds=1e9, cell_type=cell_type, cell_fault_forms=forms
    )
    injector.tally_of_form[(cell_type, "inject_fault:sigkill")] = fi.InjectionTally(
        num_attempts=num_successful_injections, num_successes=num_successful_injections
    )
    return injector


def _rollout_cell(state: fi.ObservedCellState) -> dict:
    phase = "Pending" if state is fi.ObservedCellState.PENDING else "Running"
    conditions = (
        []
        if phase == "Pending"
        else [
            {"type": "Healthy", "status": "True"},
            {"type": "Serving", "status": "True" if state is fi.ObservedCellState.SERVING else "False"},
        ]
    )
    return {
        "metadata": {"name": _ROLLOUT_CELL_NAME, "labels": {"miles.io/cell-type": "rollout"}},
        "status": {"phase": phase, "conditions": conditions},
    }


def _write_shrink_only_events(event_dir: Path) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=MainProcessIdentity())
    event_logger.log(
        CellReconfigureEvent,
        dict(rollout_id=2, quorum_id=1, src_cell_index=None, healed_cell_indices=[], alive_cell_indices_after=[0]),
        print_log=False,
    )
    event_logger.close()


class TestAssertHealing:
    def test_trainer_soak_rejects_missing_reconfigure_witness(self, tmp_path: Path) -> None:
        """A trainer-only soak whose accepted injections produced no healing event must fail."""
        _write_shrink_only_events(tmp_path / "events")

        with pytest.raises(AssertionError, match="Healing witness failed"):
            assert_healing(
                _mode("train"),
                injector=_injector(cell_type="actor", num_successful_injections=3),
                dump_dir=str(tmp_path),
            )

    def test_rollout_soak_rejects_unfinished_engine_recovery(self, tmp_path: Path) -> None:
        """A rollout-only soak that ends with an accepted injection still relaunching must fail."""
        injector = _injector(cell_type="rollout", num_successful_injections=2)
        witness = injector.recovery_witness
        witness.observe([_rollout_cell(fi.ObservedCellState.SERVING)])
        witness.note_injected(_ROLLOUT_CELL_NAME)
        witness.observe([_rollout_cell(fi.ObservedCellState.PENDING)])
        witness.observe([_rollout_cell(fi.ObservedCellState.SERVING)])
        witness.note_injected(_ROLLOUT_CELL_NAME)
        witness.observe([_rollout_cell(fi.ObservedCellState.PENDING)])

        with pytest.raises(AssertionError, match="Rollout recovery witness failed"):
            assert_healing(_mode("rollout"), injector=injector, dump_dir=str(tmp_path))


class TestAssertEveryDrawnFaultFormWorked:
    def test_a_form_that_never_worked_fails_the_soak(self, tmp_path: Path) -> None:
        """Pod deletion can be refused for the whole run while the kills alone clear the injection floor."""
        injector = _injector(cell_type="actor", num_successful_injections=3)
        injector.tally_of_form[("actor", fi.DELETE_POD_FORM_NAME)] = fi.InjectionTally(num_attempts=4, num_successes=0)

        with pytest.raises(AssertionError, match=fi.DELETE_POD_FORM_NAME):
            _assert_every_drawn_fault_form_worked(injector)

    def test_a_form_that_worked_at_least_once_is_accepted(self) -> None:
        """A single refusal is a cluster hiccup, not proof the fault form is wired up wrong."""
        injector = _injector(cell_type="actor", num_successful_injections=3)
        injector.tally_of_form[("actor", fi.DELETE_POD_FORM_NAME)] = fi.InjectionTally(num_attempts=4, num_successes=1)

        _assert_every_drawn_fault_form_worked(injector)
