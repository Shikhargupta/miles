from collections.abc import Sequence

from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    ClusterSnapshot,
    HotRestartEvidence,
    compute_trainer_boot_uuids,
)

MINIMUM_OBSERVATION_SUCCESS_RATIO: float = 0.5


def assert_the_trainer_never_rebooted(evidence: HotRestartEvidence) -> None:
    boot_uuids = compute_trainer_boot_uuids(evidence.snapshots)
    assert len(boot_uuids) == 1, (
        f"the trainer's rpc server answered with the boot uuid(s) {sorted(boot_uuids)}, and a hot restart keeps "
        f"that one process alive across every orchestration script, so anything but one uuid means the process "
        f"the take-over reloaded a checkpoint into is not the process that had been training"
    )

    for stamp, index in sorted(compute_first_snapshot_index_of_stamp(evidence.snapshots).items()):
        answered = [one for one in evidence.snapshots[index:] if one.trainer_boot_uuid is not None]
        assert answered, (
            f"the take-over stamped {stamp} and the trainer's rpc server was never reached again after it, so the "
            f"one uuid this run collected only proves the trainer was alive before anything replaced its script"
        )


def compute_first_snapshot_index_of_stamp(snapshots: Sequence[ClusterSnapshot]) -> dict[str, int]:
    first_index_of_stamp: dict[str, int] = {}
    for index, snapshot in enumerate(snapshots):
        for workload in snapshot.workloads:
            if workload.restart_at is not None:
                first_index_of_stamp.setdefault(workload.restart_at, index)
    return first_index_of_stamp


def assert_the_run_was_watched_closely_enough(
    evidence: HotRestartEvidence, *, minimum_success_ratio: float = MINIMUM_OBSERVATION_SUCCESS_RATIO
) -> None:
    observations = evidence.observations
    assert observations.attempts_of_read, (
        "nothing recorded how often this run was observed, so a verdict of 'no pod was replaced' could as well "
        "describe a cluster nobody ever managed to read"
    )

    poor = {
        read: observations.success_ratio_of(read)
        for read in sorted(observations.attempts_of_read)
        if observations.success_ratio_of(read) < minimum_success_ratio
    }
    assert not poor, (
        f"the reads {poor} answered less than {minimum_success_ratio:.0%} of the time out of "
        f"{observations.attempts_of_read}, and a run watched that badly hides the very pod restart this test is "
        f"here to catch"
    )
