from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    HotRestartEvidence,
    compute_hot_restart_workloads,
    compute_restart_stamps_of_workload,
    compute_unattributed_pod_names,
    compute_workloads_whose_pods_were_replaced,
    compute_workloads_whose_template_changed,
)


def assert_only_the_orchestration_side_restarted(evidence: HotRestartEvidence, *, num_restarts: int) -> None:
    expected = compute_hot_restart_workloads(evidence.release)

    unattributed = compute_unattributed_pod_names(evidence.snapshots)
    assert not unattributed, (
        f"the pods {sorted(unattributed)} belong to no workload this run listed, so nothing here would notice them "
        f"being replaced; every pod of the release has to be owned by a statefulset or a leaderworkerset of it"
    )

    replaced = compute_workloads_whose_pods_were_replaced(evidence.snapshots)
    assert set(replaced) == expected, (
        f"a hot restart replaces the pods of {sorted(expected)} and leaves every other pod of the run running, "
        f"and these workloads lost a pod instead: {replaced}"
    )

    rolled = compute_workloads_whose_template_changed(evidence.snapshots)
    assert rolled == expected, (
        f"only {sorted(expected)} may be rolled by a hot restart, and the pod template of {sorted(rolled)} "
        f"changed, so a relaunch that promised to leave the run's trainers and engines alone rewrote them"
    )

    stamps_of_workload = compute_restart_stamps_of_workload(evidence.snapshots)
    for workload in sorted(expected):
        assert len(stamps := stamps_of_workload[workload]) == num_restarts, (
            f"{workload} was observed carrying {sorted(stamps)}, and {num_restarts} hot restart(s) stamp one "
            f"value each, so a restart either never reached this workload or never landed"
        )
    unexpected = {
        name: sorted(stamps) for name, stamps in stamps_of_workload.items() if stamps and name not in expected
    }
    assert (
        not unexpected
    ), f"a hot restart stamps exactly the two pod templates it replaces, and these carry a stamp too: {unexpected}"
