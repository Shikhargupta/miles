import json
from typing import Any

from tests.fast.charts.utils import (
    RUN_ORCHESTRATOR_NAME,
    named_object,
    objects_of_kind,
    render_run,
    requires_helm,
    with_object_names,
)

_READER = {"name": "inference-registration-reporter", "command": ["x"], "serviceAccountName": RUN_ORCHESTRATOR_NAME}
_PLAIN = {"name": "inference-engine", "command": ["x"]}


def _render_without_orchestrator(*workers: dict[str, Any]) -> list[dict[str, Any]]:
    return render_run(
        "--set-json",
        "run.orchestrator.command=[]",
        "--set-json",
        f"run.staticWorkers={json.dumps(with_object_names(list(workers)))}",
    )


@requires_helm
class TestTheRbacThePodReadersNeed:
    def test_grants_a_release_that_runs_no_orchestration_script_but_reads_pods(self):
        """A split run's inference release carries the registration reporter, which lists pods to report them."""
        objects = _render_without_orchestrator(_READER)

        assert named_object(objects, "ServiceAccount", RUN_ORCHESTRATOR_NAME)
        assert named_object(objects, "Role", RUN_ORCHESTRATOR_NAME)
        assert named_object(objects, "RoleBinding", RUN_ORCHESTRATOR_NAME)

    def test_reads_pods_with_the_verbs_the_reporter_calls(self):
        """The reporter lists pods; a role without list would leave it forbidden at runtime."""
        role = named_object(_render_without_orchestrator(_READER), "Role", RUN_ORCHESTRATOR_NAME)

        pods = [rule for rule in role["rules"] if "pods" in rule["resources"]]
        assert [verb for verb in ("get", "list", "watch") if verb not in pods[0]["verbs"]] == []

    def test_grants_nothing_to_a_release_no_pod_reader_lives_in(self):
        """A trainer release binds no service account, so rights to delete pods would be handed to nobody."""
        objects = _render_without_orchestrator(_PLAIN)

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []

    def test_still_grants_the_release_that_runs_the_orchestration_script(self):
        """The orchestrator deletes pods to heal cells, and it is the reason this rbac existed at all."""
        assert named_object(render_run(), "ServiceAccount", RUN_ORCHESTRATOR_NAME)
