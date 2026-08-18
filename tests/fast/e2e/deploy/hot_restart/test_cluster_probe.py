import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_probe import (
    compute_trainer_rpc_url,
    parse_pod_facts,
    parse_workload_facts,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import LEADER_WORKER_SET_KIND, STATEFUL_SET_KIND, PodFact
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    NAMESPACE,
    ORCHESTRATOR,
    RELEASE,
    TRAINER,
    workload_fact,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION


class TestComputeTrainerRpcUrl:
    def test_the_trainer_is_asked_for_its_boot_uuid_at_its_own_pod(self):
        """A trainer that did restart answers the same url with a different boot uuid."""
        url = compute_trainer_rpc_url(release=RELEASE, namespace=NAMESPACE, trainer_id="actor")

        assert url.startswith("http://")
        assert NAMESPACE in url
        assert url.endswith("/v1/health")


class TestParsePodFacts:
    def test_a_pod_is_identified_by_the_uid_it_was_created_with(self):
        """A replaced pod keeps its name, so only the uid tells it apart from the one that survived."""
        payload = {
            "items": [
                {
                    "metadata": {"name": "b", "uid": "uid-b"},
                    "status": {"containerStatuses": [{"restartCount": 1}, {"restartCount": 2}]},
                },
                {"metadata": {"name": "a", "uid": "uid-a"}, "status": {}},
            ]
        }

        assert parse_pod_facts(payload) == (
            PodFact(name="a", uid="uid-a", restart_count=0),
            PodFact(name="b", uid="uid-b", restart_count=3),
        )


class TestParseWorkloadFacts:
    def test_the_generation_and_the_stamp_of_each_statefulset_are_read(self):
        """A rolled workload is one whose pod template changed, which its generation records."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ORCHESTRATOR, "generation": 2},
                    "spec": {"template": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}}},
                },
                {"metadata": {"name": TRAINER, "generation": 1}, "spec": {"template": {"metadata": {}}}},
            ]
        }

        assert parse_workload_facts(payload, kind=STATEFUL_SET_KIND) == (
            workload_fact(ORCHESTRATOR, generation=2, restart_at="t1"),
            workload_fact(TRAINER),
        )

    def test_a_leaderworkerset_carries_its_stamp_on_the_template_of_its_group(self):
        """The trainer cells and the engines of a run are leaderworkersets, not statefulsets."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ENGINE_POOL, "generation": 3},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}}
                        }
                    },
                }
            ]
        }

        assert parse_workload_facts(payload, kind=LEADER_WORKER_SET_KIND) == (
            workload_fact(ENGINE_POOL, kind=LEADER_WORKER_SET_KIND, generation=3, restart_at="t1"),
        )

    def test_an_object_stamped_twice_over_is_refused(self):
        """One stamp per replaced object is what makes counting the stamps count the take-overs."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ENGINE_POOL, "generation": 3},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "leaderTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}},
                            "workerTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t2"}}},
                        }
                    },
                }
            ]
        }

        with pytest.raises(AssertionError, match="restart stamps"):
            parse_workload_facts(payload, kind=LEADER_WORKER_SET_KIND)
