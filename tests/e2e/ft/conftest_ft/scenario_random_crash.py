# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    ModeOption,
    NumStepsOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    get_launch_plan,
    materialize_cyclic_debug_rollout_data,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.fault_injection import (
    ACTOR_CELL_TYPE,
    API_SERVER_PORT,
    ROLLOUT_CELL_TYPE,
    FaultInjectorHandle,
    compute_mean_interval_seconds_of_cell_type,
    create_cell_fault_forms,
    spawn_fault_injector,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.reconfigure_assertions import (
    assert_min_soak_injections,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)
from miles.utils.workers.naming import parse_cell_id

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_SEED: int = 42
DEFAULT_NUM_STEPS: int = 30
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 120.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 240.0

HEAL_CLOCK_SKEW_TOLERANCE: timedelta = timedelta(seconds=60)


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = DEFAULT_SEED,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    fully_async: FullyAsyncOption = False,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Starts a background thread that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    if fully_async:
        assert_mode_supports_fully_async(ft_mode, mode=mode)

    config = command_utils.default_config()
    test_name: str = f"{TEST_NAME}_fully_async" if fully_async else TEST_NAME
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}")
    print(f"Dump directory: {dump_dir}")
    mean_interval_seconds_of_cell_type: dict[str, float] = compute_mean_interval_seconds_of_cell_type(
        ft_mode.ft_components,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")
    launch_plan = get_launch_plan(fully_async=fully_async)
    print(f"Train script: {launch_plan.train_script}")

    prepare(ft_mode, config=config)

    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(ft_mode)
        + launch_plan.extra_args
        + f"--api-server-port {API_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    base_url = f"http://{config.create_backend().api_server_host()}:{API_SERVER_PORT}"
    injector = spawn_fault_injector(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config),
    )

    try:
        run_training(
            train_args=train_args,
            mode=ft_mode,
            dump_dir=dump_dir,
            config=config,
            train_script=launch_plan.train_script,
            extra_env_vars=launch_plan.env_vars,
        )
    finally:
        injector.stop_and_join()

    assert_healing(
        ft_mode.ft_components, injector=injector, event_dir=Path(dump_dir) / "events", context=f"{test_name} {mode}"
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


def assert_mode_supports_fully_async(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout, (
        f"Mode {mode!r} has no rollout engines, so a fully-async soak would train off pre-recorded debug rollout "
        f"data and would prove nothing about generating while training"
    )
    assert (
        not ft_mode.colocate
    ), f"Mode {mode!r} is colocated, which train_async.py rejects: a fully-async run needs engines of its own"


def assert_healing(
    ft_components: tuple[str, ...], *, injector: FaultInjectorHandle, event_dir: Path, context: str
) -> None:
    witness = injector.recovery_witness

    _assert_every_drawn_fault_form_worked(injector)

    if "train" in ft_components:
        assert_soak_reconfigure_events(
            event_dir, num_successful_injections=witness.num_injections(cell_type=ACTOR_CELL_TYPE)
        )
        _assert_every_trainer_injection_healed(injector, event_dir=event_dir)

    if "rollout" in ft_components:
        assert_min_soak_injections(
            witness.num_injections(cell_type=ROLLOUT_CELL_TYPE), context=f"{context} rollout cells"
        )
        assert_every_rollout_injection_recovered(injector)


def _assert_every_drawn_fault_form_worked(injector: FaultInjectorHandle) -> None:
    never_worked = injector.forms_that_never_worked()

    assert not never_worked, (
        f"Fault forms that failed every time they were drawn: {never_worked}. A soak that meets its injection "
        f"floor on the other forms would otherwise pass while this one never crashed anything "
        f"(tallies: {injector.tally_of_form})"
    )


def _assert_every_trainer_injection_healed(injector: FaultInjectorHandle, *, event_dir: Path) -> None:
    injections: list[tuple[int, datetime]] = [
        (parse_cell_id(name).cell_index, at)
        for name, at in injector.recovery_witness.injections(cell_type=ACTOR_CELL_TYPE)
    ]
    healings: list[tuple[int, datetime]] = sorted(
        (
            (cell_index, event.timestamp)
            for event in load_reconfigure_events(event_dir)
            for cell_index in event.healed_cell_indices
        ),
        key=lambda one: one[1],
    )
    unhealed: dict[int, int] = _compute_unhealed_injections(injections, healings)

    assert not unhealed, (
        f"Trainer recovery witness failed: cell index -> accepted injection(s) never healed {unhealed} when "
        f"training ended (injected {[one[0] for one in injections]}, healed {[one[0] for one in healings]} "
        f"across the events in {event_dir})"
    )

    print(
        f"Trainer recovery witness assertion passed: every one of {len(injections)} accepted injection(s) "
        f"is paired with a later healing of the same cell"
    )


def _compute_unhealed_injections(
    injections: list[tuple[int, datetime]], healings: list[tuple[int, datetime]]
) -> dict[int, int]:
    available: list[tuple[int, datetime]] = list(healings)
    unhealed: Counter[int] = Counter()
    for cell_index, injected_at in sorted(injections, key=lambda one: one[1]):
        earliest = injected_at - HEAL_CLOCK_SKEW_TOLERANCE
        match = next(
            (one for one in available if one[0] == cell_index and one[1] >= earliest),
            None,
        )
        if match is None:
            unhealed[cell_index] += 1
        else:
            available.remove(match)
    return dict(unhealed)


def assert_every_rollout_injection_recovered(injector: FaultInjectorHandle) -> None:
    witness = injector.recovery_witness
    num_injections: int = witness.num_injections(cell_type="rollout")
    num_recoveries: int = witness.num_completed_recoveries(cell_type="rollout")
    unfinished: dict[str, int] = witness.cells_with_unfinished_recovery(cell_type="rollout")
    observed: dict[str, list[str]] = {
        name: [state.value for state in states] for name, states in witness.states_of_cell_name.items()
    }

    assert not unfinished, (
        f"Rollout recovery witness failed: {unfinished} still had an accepted injection with no completed "
        f"relaunch-and-serve cycle when training ended ({num_recoveries}/{num_injections} injection(s) "
        f"recovered; observed states: {observed})"
    )
    assert num_recoveries >= num_injections, (
        f"Rollout recovery witness failed: only {num_recoveries} completed recovery(ies) for "
        f"{num_injections} accepted injection(s) (observed states: {observed})"
    )

    print(
        f"Rollout recovery witness assertion passed: {num_recoveries} completed relaunch-and-serve cycle(s) "
        f"for {num_injections} accepted injection(s)"
    )


if __name__ == "__main__":
    app()
