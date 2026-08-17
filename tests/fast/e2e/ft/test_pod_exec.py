import subprocess

import pytest
from tests.e2e.ft.conftest_ft import pod_exec

_NAMESPACE = "miles-e2e"
_POD_NAME = "rollout-engine-0-0"
_PATTERN = "sglang::"


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _fake_kubectl(monkeypatch: pytest.MonkeyPatch, respond) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
        calls.append(argv)
        return respond(argv)

    monkeypatch.setattr(pod_exec, "run_process", fake_run_process)
    return calls


class TestSigkillProcessPatternsInPod:
    def test_it_signals_a_process_inside_the_pod_rather_than_deleting_it(self, monkeypatch: pytest.MonkeyPatch):
        """Deleting the pod is a different fault: this one leaves the pod and crashes what runs in it."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed())

        pod_exec.sigkill_process_patterns_in_pod(
            namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
        )

        assert calls[0][:2] == ["kubectl", "exec"]
        assert _PATTERN in calls[0]
        assert "delete" not in calls[0]

    def test_it_names_the_container_it_reaches_into(self, monkeypatch: pytest.MonkeyPatch):
        """A pod can hold sidecars, and killing a process in the wrong one is not the fault under test."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed())

        pod_exec.sigkill_process_patterns_in_pod(
            namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
        )

        assert calls[0][calls[0].index("--container") + 1] == "engine"

    def test_matching_no_process_is_a_failure_rather_than_a_crash_nobody_caused(self, monkeypatch: pytest.MonkeyPatch):
        """pkill exits 1 when it matched nothing, which is indistinguishable from a kill that worked."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(returncode=1))

        with pytest.raises(AssertionError, match="No process matching"):
            pod_exec.sigkill_process_patterns_in_pod(
                namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
            )
