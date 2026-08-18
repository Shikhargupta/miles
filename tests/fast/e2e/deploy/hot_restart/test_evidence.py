import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    BOOT_UUID_READ,
    LEADER_WORKER_SET_KIND,
    POD_KIND,
    HotRestartEvidence,
    HotRestartRecord,
    ObservationCounts,
    compute_hot_restart_workloads,
    compute_restart_stamps_of_workload,
    compute_trainer_boot_uuids,
    compute_unattributed_pod_names,
    compute_workload_of_pod,
    compute_workloads_whose_pods_were_replaced,
    compute_workloads_whose_template_changed,
)
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    ORCHESTRATOR,
    RELEASE,
    ROLLOUT_EXECUTOR,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)


class TestComputeHotRestartWorkloads:
    def test_a_hot_restart_names_the_orchestrator_and_the_rollout_executor(self):
        """Every other object of the release has to come out of the relaunch untouched."""
        assert compute_hot_restart_workloads(RELEASE) == frozenset({ORCHESTRATOR, ROLLOUT_EXECUTOR})


class TestComputeWorkloadOfPod:
    def test_a_pod_belongs_to_the_workload_whose_name_it_extends(self):
        """Pods are named after their statefulset plus an ordinal, and nothing else links them."""
        assert compute_workload_of_pod(f"{TRAINER}-0", workloads=[TRAINER, ORCHESTRATOR]) == TRAINER

    def test_the_longest_matching_workload_wins(self):
        """One workload's name can prefix another's, and the shorter one would then claim its pods."""
        assert (
            compute_workload_of_pod(
                f"{RELEASE}-router-extra-0", workloads=[f"{RELEASE}-router", f"{RELEASE}-router-extra"]
            )
            == f"{RELEASE}-router-extra"
        )

    def test_a_worker_of_a_leaderworkerset_group_belongs_to_that_leaderworkerset(self):
        """A group's workers are named after the set, the group index and the worker index."""
        assert compute_workload_of_pod(f"{ENGINE_POOL}-1-2", workloads=[ENGINE_POOL]) == ENGINE_POOL

    def test_a_pod_of_no_known_workload_belongs_to_none(self):
        """Only the pods of this release's workloads say anything about this run."""
        assert compute_workload_of_pod("unrelated-0", workloads=[TRAINER]) is None


class TestComputeUnattributedPodNames:
    def test_a_run_whose_every_pod_belongs_to_a_known_workload_leaves_the_bucket_empty(self):
        """A pod nothing owns is a pod no restart verdict covers."""
        snapshot = cluster_snapshot(
            pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-e"), pod_fact(f"{TRAINER}-0", uid="uid-t")],
            workloads=[workload_fact(ENGINE_POOL, kind=LEADER_WORKER_SET_KIND), workload_fact(TRAINER)],
        )

        assert compute_unattributed_pod_names([snapshot]) == set()

    def test_a_pod_of_a_workload_kind_nobody_listed_is_reported(self):
        """The engines were invisible to the verdict for as long as only statefulsets were read."""
        snapshot = cluster_snapshot(
            pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-e")], workloads=[workload_fact(TRAINER)]
        )

        assert compute_unattributed_pod_names([snapshot]) == {f"{ENGINE_POOL}-0-1"}

    def test_an_observation_that_could_not_list_every_workload_is_not_read_as_an_orphan_pod(self):
        """A failed kubectl call says nothing about which workload a pod belongs to."""
        snapshot = cluster_snapshot(
            pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-e")],
            workloads=[workload_fact(TRAINER)],
            reads_missing=(LEADER_WORKER_SET_KIND,),
        )

        assert compute_unattributed_pod_names([snapshot]) == set()


class TestComputeWorkloadsWhosePodsWereReplaced:
    def test_a_run_nothing_disturbed_replaces_no_pod(self):
        """This is what every workload outside the hot restart has to look like."""
        snapshot = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)])

        assert compute_workloads_whose_pods_were_replaced([snapshot, snapshot]) == {}

    def test_a_pod_that_came_back_under_a_new_uid_names_its_workload(self):
        """A rolled statefulset recreates its pod under the same name, so only the uid shows it."""
        workloads = [workload_fact(ORCHESTRATOR), workload_fact(TRAINER)]
        before = cluster_snapshot(
            pods=[pod_fact(f"{ORCHESTRATOR}-0", uid="uid-1"), pod_fact(f"{TRAINER}-0", uid="uid-t")],
            workloads=workloads,
        )
        after = cluster_snapshot(
            pods=[pod_fact(f"{ORCHESTRATOR}-0", uid="uid-2"), pod_fact(f"{TRAINER}-0", uid="uid-t")],
            workloads=workloads,
        )

        assert list(compute_workloads_whose_pods_were_replaced([before, after])) == [ORCHESTRATOR]

    def test_an_engine_pod_that_came_back_names_the_leaderworkerset_it_belongs_to(self):
        """The engines only take part in the verdict once their leaderworkerset is known."""
        workloads = [workload_fact(ENGINE_POOL, kind=LEADER_WORKER_SET_KIND)]
        before = cluster_snapshot(pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-1")], workloads=workloads)
        after = cluster_snapshot(pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-2")], workloads=workloads)

        assert list(compute_workloads_whose_pods_were_replaced([before, after])) == [ENGINE_POOL]

    def test_a_container_that_restarted_in_place_names_its_workload(self):
        """A trainer whose container crashed and came back lost the state a take-over relies on."""
        workloads = [workload_fact(TRAINER)]
        before = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=workloads)
        after = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t", restart_count=1)], workloads=workloads)

        assert list(compute_workloads_whose_pods_were_replaced([before, after])) == [TRAINER]

    def test_a_pod_that_disappeared_names_its_workload(self):
        """A pod deleted and not yet recreated is a restart caught mid-flight, not a run left alone."""
        workloads = [workload_fact(TRAINER)]
        before = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=workloads)
        after = cluster_snapshot(pods=[], workloads=workloads)

        assert list(compute_workloads_whose_pods_were_replaced([before, after])) == [TRAINER]


class TestComputeWorkloadsWhoseTemplateChanged:
    def test_a_relaunch_that_rolled_nothing_changes_no_generation(self):
        """An ordinary relaunch has to render a zero diff, or it would restart the whole run."""
        snapshot = cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER), workload_fact(ORCHESTRATOR)])

        assert compute_workloads_whose_template_changed([snapshot, snapshot]) == set()

    def test_only_the_workloads_whose_generation_moved_are_reported(self):
        """A hot restart stamps two pod templates, and the generation is how kubernetes records that."""
        before = cluster_snapshot(pods=[], workloads=[workload_fact(ORCHESTRATOR), workload_fact(TRAINER)])
        after = cluster_snapshot(
            pods=[], workloads=[workload_fact(ORCHESTRATOR, generation=2), workload_fact(TRAINER)]
        )

        assert compute_workloads_whose_template_changed([before, after]) == {ORCHESTRATOR}

    def test_a_workload_restamped_between_two_observations_of_one_generation_is_reported(self):
        """An observation may miss the generation moving, but the stamp it carries is written to survive."""
        before = cluster_snapshot(pods=[], workloads=[workload_fact(ENGINE_POOL, generation=2, restart_at="t1")])
        after = cluster_snapshot(pods=[], workloads=[workload_fact(ENGINE_POOL, generation=2, restart_at="t2")])

        assert compute_workloads_whose_template_changed([before, after]) == {ENGINE_POOL}


class TestComputeRestartStampsOfWorkload:
    def test_every_distinct_stamp_a_workload_carried_is_collected(self):
        """Each hot restart writes one stamp, so the count of them is the count of take-overs."""
        snapshots = [
            cluster_snapshot(pods=[], workloads=[workload_fact(ORCHESTRATOR), workload_fact(TRAINER)]),
            cluster_snapshot(
                pods=[], workloads=[workload_fact(ORCHESTRATOR, restart_at="t1"), workload_fact(TRAINER)]
            ),
            cluster_snapshot(
                pods=[], workloads=[workload_fact(ORCHESTRATOR, restart_at="t2"), workload_fact(TRAINER)]
            ),
        ]

        assert compute_restart_stamps_of_workload(snapshots) == {ORCHESTRATOR: {"t1", "t2"}, TRAINER: set()}


class TestComputeTrainerBootUuids:
    def test_a_trainer_that_outlived_every_script_answers_with_one_boot_uuid(self):
        """A second uuid means the process a hot restart promised to keep alive was replaced."""
        snapshots = [
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid=None),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
        ]

        assert compute_trainer_boot_uuids(snapshots) == {"boot-a"}


class TestObservationCounts:
    def test_a_read_that_never_answered_has_a_success_ratio_of_zero(self):
        """A verdict is only worth what the observations behind it cost to collect."""
        counts = ObservationCounts(attempts_of_read={POD_KIND: 4}, failures_of_read={POD_KIND: 3})

        assert counts.success_ratio_of(POD_KIND) == 0.25
        assert counts.success_ratio_of(BOOT_UUID_READ) == 0.0


class TestHotRestartEvidence:
    def test_what_the_target_side_observed_survives_to_the_comparison(self, tmp_path):
        """The compare step runs as a subcommand of its own and cannot watch the run itself."""
        evidence = HotRestartEvidence(
            records=(HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=2),),
            snapshots=(
                cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)]),
            ),
            release=RELEASE,
            observations=ObservationCounts(attempts_of_read={POD_KIND: 2}, failures_of_read={POD_KIND: 1}),
        )
        evidence.write(dump_dir=str(tmp_path))

        assert HotRestartEvidence.load(dump_dir=str(tmp_path)) == evidence

    def test_comparing_without_a_target_run_fails_loudly(self, tmp_path):
        """A missing file would otherwise read as a run that redid nothing."""
        with pytest.raises(AssertionError):
            HotRestartEvidence.load(dump_dir=str(tmp_path))
