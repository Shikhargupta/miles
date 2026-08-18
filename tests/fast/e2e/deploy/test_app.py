from dataclasses import dataclass, field

import pytest
from tests.e2e.deploy.conftest_deploy import app as deploy_app
from tests.e2e.deploy.conftest_deploy import split_deployment
from tests.e2e.deploy.conftest_deploy.app import create_split_comparison_app_and_run_ci
from tests.e2e.deploy.conftest_deploy.split_deployment import RunDeployment
from tests.e2e.ft.conftest_ft import app as ft_app
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
TEST_NAME: str = "split_dispatch"

MODE: FTTestMode = FTTestMode(
    model_name="Qwen3-0.6B",
    model_hf_repo="Qwen/Qwen3-0.6B",
    megatron_model_type="qwen3-0.6B",
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)


class TestCreateSplitComparisonAppAndRunCi:
    def test_only_the_target_side_is_installed_as_several_deployments(self, pipeline):
        """A baseline that was also split would compare two split runs and prove nothing about splitting."""
        pipeline.run_ci()

        assert pipeline.split_sides == [TARGET_SIDE]
        assert pipeline.unsplit_sides == [BASELINE_SIDE]

    def test_the_run_reaches_the_cluster_one_deployment_at_a_time_in_this_order(self, pipeline):
        """The baseline is one release; the target's parts install in the order the run needs them installed."""
        pipeline.run_ci()

        assert pipeline.launched == [
            DeployComponent.ALL,
            DeployComponent.TRAINER,
            DeployComponent.INFERENCE,
            DeployComponent.INFERENCE,
            DeployComponent.PRIMARY,
        ]

    def test_every_engine_deployment_of_the_target_is_named_apart_from_the_others(self, pipeline):
        """Two engine deployments installed under one name leave the run with half the engines it counted on."""
        pipeline.run_ci()

        assert pipeline.instance_ids == [None, None, "e0", "e1", None]

    def test_the_two_sides_are_compared_once_both_have_run(self, pipeline):
        """A comparison run before the second side would read one side's dumps as both."""
        pipeline.run_ci()

        assert pipeline.compared_after == len(pipeline.launched)

    def test_a_cluster_that_cannot_carry_the_releases_deploys_nothing(self, pipeline):
        """An environment without kubernetes has to skip, not install half a run and fail deep inside helm."""
        pipeline.cluster_is_ready = False

        pipeline.run_ci()

        assert not pipeline.launched


@dataclass
class _Pipeline:
    launched: list[DeployComponent] = field(default_factory=list)
    instance_ids: list[str | None] = field(default_factory=list)
    split_sides: list[str] = field(default_factory=list)
    unsplit_sides: list[str] = field(default_factory=list)
    compared_after: int | None = None
    cluster_is_ready: bool = True

    def run_ci(self) -> None:
        _, run_ci = create_split_comparison_app_and_run_ci(
            test_name=TEST_NAME,
            mode=MODE,
            build_baseline_args=_build_baseline_args,
            build_target_args=_build_args,
            build_deployments=self.build_deployments,
            compare_fn=self.compare,
        )
        run_ci()

    def build_deployments(self, request: RunSideRequest) -> list[RunDeployment]:
        self.split_sides.append(request.side)
        return [
            RunDeployment(deploy_component=DeployComponent.TRAINER, train_args=request.train_args),
            RunDeployment(
                deploy_component=DeployComponent.INFERENCE, train_args=request.train_args, deploy_instance_id="e0"
            ),
            RunDeployment(
                deploy_component=DeployComponent.INFERENCE, train_args=request.train_args, deploy_instance_id="e1"
            ),
            RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args=request.train_args),
        ]

    def compare(self, dump_dir: str, mode: FTTestMode) -> None:
        self.compared_after = len(self.launched)

    def record(self, config: ExecuteTrainConfig) -> None:
        self.launched.append(config.deploy_component)
        self.instance_ids.append(config.deploy_instance_id)


def _build_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return "--some-flag some-value "


def _build_baseline_args(request: RunSideRequest) -> str:
    return "--some-flag some-value "


@pytest.fixture
def pipeline(monkeypatch, tmp_path) -> _Pipeline:
    recorded = _Pipeline()

    def run_unsplit(request: RunSideRequest) -> None:
        recorded.unsplit_sides.append(request.side)
        ft_app.run_one_release(request)

    monkeypatch.setattr(deploy_app, "is_cluster_ready_for_helm_runs", lambda config: recorded.cluster_is_ready)
    monkeypatch.setattr(ft_app, "resolve_dump_dir", lambda test_name: str(tmp_path / test_name))
    monkeypatch.setattr(ft_app, "prepare", lambda mode: None)
    monkeypatch.setattr(command_utils, "default_config", _config)
    monkeypatch.setattr(
        ft_app, "run_training", lambda *, train_args, mode, dump_dir=None, config: recorded.record(config)
    )
    monkeypatch.setattr(split_deployment, "run_training", lambda *, train_args, mode, config: recorded.record(config))
    monkeypatch.setattr(split_deployment, "run_one_release", run_unsplit)
    monkeypatch.setattr(split_deployment, "Helm", _FakeHelm())

    return recorded


def _config() -> ExecuteTrainConfig:
    return ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID)


class _FakeHelm:
    @staticmethod
    def get_manifest(release: str, namespace: str) -> object | None:
        return object()

    @staticmethod
    def uninstall(*, release: str, namespace: str) -> None:
        return None
