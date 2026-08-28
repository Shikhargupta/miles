import json
from typing import Any

from tests.fast.charts.utils import (
    RUN_ORCHESTRATOR_NAME,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    requires_helm,
    with_object_names,
)

_READER = {
    "name": "inference-registration-reporter",
    "command": ["x"],
    "serviceAccountName": RUN_ORCHESTRATOR_NAME,
}
_PLAIN = {"name": "inference-engine", "command": ["x"]}


def _render_without_orchestrator(*workers: dict[str, Any]) -> list[dict[str, Any]]:
    return render_run(
        "--set-json",
        "run.orchestrator.command=[]",
        "--set-json",
        f"run.staticWorkers={json.dumps(with_object_names(list(workers)))}",
    )


@requires_helm
class TestTheRbacAReleaseWithoutAnOrchestrationScriptNeeds:
    def test_grants_the_account_a_release_binds_its_pod_readers_to(self):
        """A split run's inference release carries the registration reporter, which lists pods to report them."""
        objects = _render_without_orchestrator(_READER)

        assert named_object(objects, "ServiceAccount", RUN_ORCHESTRATOR_NAME)
        assert named_object(objects, "Role", RUN_ORCHESTRATOR_NAME)
        assert named_object(objects, "RoleBinding", RUN_ORCHESTRATOR_NAME)
        assert (
            pod_spec_of(objects, "StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-inference-registration-reporter")[
                "serviceAccountName"
            ]
            == RUN_ORCHESTRATOR_NAME
        )

    def test_grants_nothing_to_a_release_no_pod_reader_lives_in(self):
        """A trainer release binds no service account, so rights over pods would be handed to nobody."""
        objects = _render_without_orchestrator(_PLAIN)

        assert objects_of_kind(objects, "ServiceAccount") == []
        assert objects_of_kind(objects, "Role") == []
        assert objects_of_kind(objects, "RoleBinding") == []

    def test_still_grants_the_release_that_runs_the_orchestration_script(self):
        """The orchestrator deletes pods to heal cells, and it is the reason this rbac existed at all."""
        objects = render_run()

        assert named_object(objects, "ServiceAccount", RUN_ORCHESTRATOR_NAME)
        assert named_object(objects, "Role", RUN_ORCHESTRATOR_NAME)["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]
