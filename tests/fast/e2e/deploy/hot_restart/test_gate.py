import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.gate import (
    GateStage,
    NoCheckpointGate,
    RestartGate,
    compute_next_gate,
    compute_next_no_checkpoint_gate,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import RunProgress


class TestRestartGate:
    def test_a_run_that_has_not_saved_keeps_the_gate_shut(self):
        """A take-over before the first save would resume from the reference weights, not from this run."""
        gate = RestartGate()

        assert not gate.observe(RunProgress(last_saved_iteration=None, last_finished_rollout_id=4))
        assert gate.stage is GateStage.AWAITING_SAVE

    def test_a_save_alone_keeps_the_gate_shut(self):
        """Restarting right after a save would waste no step, so nothing would prove a step is redone."""
        gate = RestartGate()

        assert not gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))
        assert gate.stage is GateStage.AWAITING_STEP

    def test_a_save_seen_before_any_step_was_read_is_not_taken_as_a_step_of_its_own(self):
        """Latching no step at all would let the very next step open the gate on an empty window."""
        gate = RestartGate()

        assert not gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=None))
        assert gate.stage is GateStage.AWAITING_SAVE
        assert not gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))
        assert gate.stage is GateStage.AWAITING_STEP

    def test_a_step_after_the_save_opens_the_gate(self):
        """This is the whole precondition: a checkpoint to resume from, and work past it to redo."""
        gate = RestartGate()
        gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))

        assert gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))
        assert gate.stage is GateStage.OPEN

    def test_a_step_finished_before_the_save_was_seen_does_not_open_the_gate(self):
        """The step that has to be wasted is one that ran after the checkpoint, not one it contains."""
        gate = RestartGate()
        gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=3))

        assert not gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=3))

    def test_a_later_restart_waits_for_a_checkpoint_past_the_steps_the_previous_one_wasted(self):
        """Two restarts sharing a redone step would make one step cost three attempts, not two."""
        gate = RestartGate(minimum_saved_iteration=5)

        assert not gate.observe(RunProgress(last_saved_iteration=4, last_finished_rollout_id=9))
        assert gate.stage is GateStage.AWAITING_SAVE
        assert not gate.observe(RunProgress(last_saved_iteration=5, last_finished_rollout_id=9))
        assert gate.observe(RunProgress(last_saved_iteration=5, last_finished_rollout_id=10))


class TestComputeNextGate:
    def test_the_first_restart_waits_for_any_checkpoint(self):
        """Nothing has been redone yet, so every save is far enough along."""
        assert compute_next_gate([]).minimum_saved_iteration is None

    def test_every_later_restart_waits_past_the_last_step_the_previous_one_redid(self):
        """The windows two restarts redo have to be disjoint for the attempt count to be readable."""
        record = HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=3)

        assert compute_next_gate([record]).minimum_saved_iteration == 3


class TestTheRecordARestartGateComputes:
    def test_the_record_is_what_the_run_had_reached_when_the_restart_was_triggered(self):
        """A take-over lands seconds later, so this records the trigger, not the window that followed."""
        gate = RestartGate()
        gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))
        progress = RunProgress(last_saved_iteration=2, last_finished_rollout_id=3)
        gate.observe(progress)

        record = gate.compute_record(index=0, progress=progress)

        assert record == HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=3)

    def test_a_gate_that_never_opened_records_nothing(self):
        """Recording a restart that was not due would claim steps were redone that never ran twice."""
        with pytest.raises(AssertionError):
            RestartGate().compute_record(
                index=0, progress=RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
            )


class TestNoCheckpointGate:
    def test_a_run_that_has_finished_no_step_keeps_the_gate_shut(self):
        """A take-over before the first step would redo nothing and prove nothing about redoing anything."""
        gate = NoCheckpointGate()

        assert not gate.observe(RunProgress(last_saved_iteration=None, last_finished_rollout_id=None))
        assert gate.stage is GateStage.AWAITING_STEP

    def test_one_finished_step_of_a_run_that_has_saved_nothing_opens_the_gate(self):
        """This is the whole precondition: work to throw away, and no checkpoint to resume from."""
        gate = NoCheckpointGate()

        assert gate.observe(RunProgress(last_saved_iteration=None, last_finished_rollout_id=0))
        assert gate.stage is GateStage.OPEN

    def test_a_save_that_lands_before_the_restart_could_fire_fails_instead_of_waiting(self):
        """Waiting on it would silently restart a run that has a checkpoint, which another scenario covers."""
        gate = NoCheckpointGate()

        with pytest.raises(AssertionError, match="before anything took it over"):
            gate.observe(RunProgress(last_saved_iteration=1, last_finished_rollout_id=1))

    def test_a_gate_that_opened_stays_open_once_the_restart_makes_the_run_save(self):
        """The take-over is already in flight by then, and the save it triggers is not a gate violation."""
        gate = NoCheckpointGate()
        gate.observe(RunProgress(last_saved_iteration=None, last_finished_rollout_id=0))

        assert gate.observe(RunProgress(last_saved_iteration=4, last_finished_rollout_id=4))

    def test_the_record_says_that_no_checkpoint_existed_when_the_restart_was_triggered(self):
        """Only the record tells the comparison which of the two take-over paths this dump describes."""
        gate = NoCheckpointGate()
        progress = RunProgress(last_saved_iteration=None, last_finished_rollout_id=2)
        gate.observe(progress)

        record = gate.compute_record(index=0, progress=progress)

        assert record == HotRestartRecord(index=0, saved_iteration_at_trigger=None, finished_rollout_id_at_trigger=2)

    def test_a_gate_that_never_opened_records_nothing(self):
        """Recording a restart that was not due would claim a step was redone that never ran twice."""
        with pytest.raises(AssertionError, match="still AWAITING_STEP"):
            NoCheckpointGate().compute_record(
                index=0, progress=RunProgress(last_saved_iteration=None, last_finished_rollout_id=2)
            )


class TestComputeNextNoCheckpointGate:
    def test_the_first_restart_of_a_run_that_has_saved_nothing_is_allowed(self):
        """Nothing has taken this run over yet, so it can still be holding no checkpoint."""
        assert compute_next_no_checkpoint_gate([]).stage is GateStage.AWAITING_STEP

    def test_a_second_restart_of_the_same_run_is_refused(self):
        """The take-over makes the run save, so no later restart finds it without a checkpoint again."""
        record = HotRestartRecord(index=0, saved_iteration_at_trigger=None, finished_rollout_id_at_trigger=2)

        with pytest.raises(AssertionError, match="restarted once"):
            compute_next_no_checkpoint_gate([record])
