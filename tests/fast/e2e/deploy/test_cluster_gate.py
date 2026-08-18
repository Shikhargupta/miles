import pytest
from tests.e2e.deploy.conftest_deploy import cluster_gate

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


def _config(*, cluster_backend: ClusterBackend, namespace: str) -> ExecuteTrainConfig:
    return ExecuteTrainConfig(cluster_backend=cluster_backend, namespace=namespace, run_id="demo")


class TestComputeUnconfiguredReason:
    def test_an_environment_on_another_backend_declares_no_kubernetes(self):
        """Only kubernetes installs one release per deployment, so any other backend is a declared absence."""
        reason = cluster_gate._compute_unconfigured_reason(_config(cluster_backend=ClusterBackend.RAY, namespace="rl"))

        assert reason is not None and ClusterBackend.RAY.value in reason

    def test_a_kubernetes_environment_without_a_namespace_is_unconfigured(self):
        """An empty namespace is how the environment says it configured no cluster of its own."""
        reason = cluster_gate._compute_unconfigured_reason(
            _config(cluster_backend=ClusterBackend.KUBERNETES, namespace="")
        )

        assert reason is not None and cluster_gate.RUN_NAMESPACE_ENV_VAR in reason

    def test_a_configured_kubernetes_environment_is_not_excused(self):
        """A declared namespace commits the environment to actually running these tests."""
        assert (
            cluster_gate._compute_unconfigured_reason(
                _config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl")
            )
            is None
        )


class TestIsClusterReadyForHelmRuns:
    def test_it_skips_without_probing_when_the_environment_declares_no_kubernetes(self, monkeypatch, capsys):
        """An unconfigured environment is skipped with its reason, and no probe is attempted."""
        monkeypatch.setattr(cluster_gate, "create_backend_for_run", _refuse_to_probe)

        ready = cluster_gate.is_cluster_ready_for_helm_runs(
            _config(cluster_backend=ClusterBackend.RAY, namespace="rl")
        )

        assert not ready
        assert "SKIPPED" in capsys.readouterr().out

    def test_a_declared_cluster_that_cannot_be_reached_fails_rather_than_skips(self, monkeypatch):
        """Exiting 0 here would report green for a test that never installed anything."""
        monkeypatch.setattr(cluster_gate, "create_backend_for_run", _refuse_to_probe)

        with pytest.raises(AssertionError):
            cluster_gate.is_cluster_ready_for_helm_runs(
                _config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl")
            )

    def test_a_reachable_cluster_runs_the_test(self, monkeypatch):
        """The gate exists to skip unconfigured environments, not to narrow configured ones."""
        monkeypatch.setattr(cluster_gate, "create_backend_for_run", lambda config: None)

        assert cluster_gate.is_cluster_ready_for_helm_runs(
            _config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl")
        )


def _refuse_to_probe(config: ExecuteTrainConfig) -> None:
    raise AssertionError(f"the {config.cluster_backend.value} backend is not reachable")
