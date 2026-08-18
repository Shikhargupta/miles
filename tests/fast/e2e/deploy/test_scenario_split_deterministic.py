import pytest
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID, RunAddressBook
from tests.e2e.deploy.conftest_deploy import scenario_split_deterministic as scenario
from tests.e2e.deploy.conftest_deploy.split_deployment import RunDeployment
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.fast.train_args import shared_argv, value_of, values_after

from miles.ray.specs.inference import INFERENCE_CONTROLLER_ADDR_FLAG
from miles.ray.specs.train import TRAINER_CONTROLLER_ADDRS_FLAG
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, OBJECT_STORE_BACKEND_FLAG
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"
DUMP_DIR: str = "/dumps/one-side"


@pytest.fixture
def mode() -> FTTestMode:
    return scenario._MODE


@pytest.fixture
def address_book() -> RunAddressBook:
    return RunAddressBook(run_id=RUN_ID, run_uuid=RUN_UUID, namespace=NAMESPACE)


@pytest.fixture
def deployments(mode: FTTestMode) -> list[RunDeployment]:
    return scenario._build_deployments(_request(mode))


class TestMode:
    def test_the_scenario_deploys_more_than_one_group_of_engines(self, mode):
        """A single engine deployment is a shape the unsplit baseline already covers."""
        assert mode.rollout_num_engines > 1

    def test_the_scenario_shares_no_gpus_between_the_trainer_and_the_engines(self, mode):
        """Separate deployments cannot be colocated, and a mode that asked for it would fail deep inside helm."""
        assert not mode.colocate


class TestBuildDeployments:
    def test_the_engines_of_a_run_are_deployed_one_group_at_a_time(self, deployments, mode):
        """Several engine deployments registering into one run is the whole subject of this scenario."""
        assert len(_deployments_of(deployments, DeployComponent.INFERENCE)) == mode.rollout_num_engines

    def test_every_engine_deployment_is_named_apart_from_the_others(self, deployments):
        """Two engine deployments of one name install one release, and the run loses half its engines."""
        instance_ids = [one.deploy_instance_id for one in _deployments_of(deployments, DeployComponent.INFERENCE)]

        assert None not in instance_ids
        assert len(set(instance_ids)) == len(instance_ids)

    def test_the_engine_deployments_together_carry_what_the_baseline_runs_alone(self, deployments, mode):
        """A target with fewer engines than the baseline would be compared against a different run."""
        declared = [
            int(value_of(one.train_args, scenario.ROLLOUT_NUM_GPUS_FLAG))
            for one in _deployments_of(deployments, DeployComponent.INFERENCE)
        ]

        assert sum(declared) == mode.total_rollout_gpus

    def test_an_engine_deployment_is_given_the_gpus_of_exactly_one_engine(self, deployments, mode):
        """A deployment holding two engines is the shape the baseline already runs, and splits nothing."""
        for one in _deployments_of(deployments, DeployComponent.INFERENCE):
            assert value_of(one.train_args, scenario.ROLLOUT_NUM_GPUS_FLAG) == str(mode.rollout_gpus_per_engine)
            assert value_of(one.train_args, scenario.ROLLOUT_NUM_GPUS_PER_ENGINE_FLAG) == str(
                mode.rollout_gpus_per_engine
            )

    def test_the_driving_deployment_still_counts_every_engine_the_run_registers(self, deployments, mode):
        """It waits for as many engine cells as its own arguments declare, and would start on half a fleet."""
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]

        assert value_of(driver.train_args, scenario.ROLLOUT_NUM_GPUS_FLAG) == str(mode.total_rollout_gpus)

    def test_only_the_engine_deployments_are_told_where_to_register(self, deployments):
        """Every other deployment holds the controller itself and refuses to be pointed at one."""
        told = {one.deploy_component for one in deployments if INFERENCE_CONTROLLER_ADDR_FLAG in one.train_args}

        assert told == {DeployComponent.INFERENCE}

    def test_every_engine_deployment_registers_into_the_controller_of_this_run(self, deployments, address_book):
        """An address that only looks right is an engine deployment that joins some other run, or none."""
        expected = value_of(address_book.inference_controller_addr_arg(), INFERENCE_CONTROLLER_ADDR_FLAG)

        for one in _deployments_of(deployments, DeployComponent.INFERENCE):
            assert value_of(one.train_args, INFERENCE_CONTROLLER_ADDR_FLAG) == expected

    def test_only_the_driving_deployment_is_told_where_the_trainer_is(self, deployments):
        """The deployment that carries the trainer reaches it in its own process and refuses the flag."""
        told = {one.deploy_component for one in deployments if TRAINER_CONTROLLER_ADDRS_FLAG in one.train_args}

        assert told == {DeployComponent.PRIMARY}

    def test_the_driving_deployment_dials_the_trainer_release_of_this_run(self, deployments, address_book):
        """The driver reaches the trainer by name alone, so a name that drifted reaches nothing at all."""
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]
        expected = address_book.trainer_controller_addrs_arg(
            deploy_instance_id_of_trainer_id={DEFAULT_TRAINER_ID: None}
        )

        assert values_after(driver.train_args, TRAINER_CONTROLLER_ADDRS_FLAG) == values_after(
            expected, TRAINER_CONTROLLER_ADDRS_FLAG
        )

    def test_the_run_is_driven_by_the_deployment_installed_last(self, deployments):
        """Installing it earlier would block on a run whose workers are not there yet."""
        assert deployments[-1].deploy_component is DeployComponent.PRIMARY

    def test_the_trainer_is_deployed_before_the_script_that_drives_it(self, deployments):
        """The driving deployment is handed the trainer's address, so the trainer has to be on its way."""
        components = [one.deploy_component for one in deployments]

        assert components.index(DeployComponent.TRAINER) < components.index(DeployComponent.PRIMARY)

    def test_every_deployment_redeems_its_references_at_one_object_store(self, deployments):
        """Deployments that disagree on the master hand each other references nothing can read back."""
        addresses = {value_of(one.train_args, MOONCAKE_INIT_KWARGS_FLAG) for one in deployments}

        assert len(addresses) == 1
        assert all(value_of(one.train_args, OBJECT_STORE_BACKEND_FLAG) == "mooncake" for one in deployments)

    def test_the_deployments_agree_on_everything_the_run_itself_declares(self, deployments):
        """Only what a deployment carries may differ; a drifted model or batch shape trains something else."""
        shared = [_shared_argv(one.train_args) for one in deployments]

        assert all(one == shared[0] for one in shared)


class TestBuildArgs:
    def test_the_baseline_installs_a_run_that_runs_its_own_object_store(self, mode):
        """The baseline is one release, so nothing outside it names the master it should dial."""
        baseline = scenario._build_baseline_args(_request(mode, side=BASELINE_SIDE))

        assert value_of(baseline, OBJECT_STORE_BACKEND_FLAG) == "mooncake"

    def test_the_two_sides_differ_in_nothing_but_how_they_are_deployed(self, mode, deployments):
        """A bitwise comparison across a second difference would prove nothing about deployment."""
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]

        baseline = scenario._build_baseline_args(_request(mode, side=BASELINE_SIDE))

        assert _shared_argv(driver.train_args) == _shared_argv(baseline)

    def test_the_baseline_is_grouped_under_the_run_it_is_launched_as(self, mode, monkeypatch):
        """A group named after a config nobody launched files the baseline's metrics under a run that never ran."""
        monkeypatch.setenv("WANDB_API_KEY", "unused-in-this-test")
        monkeypatch.delenv("GITHUB_COMMIT_NAME", raising=False)

        baseline = scenario._build_baseline_args(_request(mode, side=BASELINE_SIDE))

        assert value_of(baseline, "--wandb-group") == RUN_ID

    def test_the_run_trains_without_weight_decay(self, mode):
        """Weight decay moves weights on its own, which would let a run that learned nothing pass the moved gate."""
        assert value_of(scenario._build_args(mode, DUMP_DIR), "--weight-decay") == "0"

    def test_a_colocated_mode_is_refused(self, mode):
        """Colocation shares gpus between the very deployments this scenario installs apart."""
        with pytest.raises(AssertionError, match="colocated"):
            scenario._build_args(_colocated(mode), DUMP_DIR)

    def test_a_mode_without_engines_is_refused(self, mode):
        """There would be no engine deployment left to install, and the scenario would test nothing."""
        with pytest.raises(AssertionError, match="engines to deploy"):
            scenario._build_args(_without_engines(mode), DUMP_DIR)


def _deployments_of(deployments: list[RunDeployment], component: DeployComponent) -> list[RunDeployment]:
    return [one for one in deployments if one.deploy_component is component]


def _request(mode: FTTestMode, *, side: str = TARGET_SIDE) -> RunSideRequest:
    return RunSideRequest(
        side=side,
        mode=mode,
        train_args=scenario._build_args(mode, DUMP_DIR),
        dump_dir=DUMP_DIR,
        config=ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID
        ),
        enable_dumper=True,
    )


def _colocated(mode: FTTestMode) -> FTTestMode:
    return FTTestMode(
        model_name=mode.model_name,
        model_hf_repo=mode.model_hf_repo,
        megatron_model_type=mode.megatron_model_type,
        num_cells=mode.num_cells,
        train_gpus_per_node=mode.train_gpus_per_node,
        rollout_num_engines=mode.rollout_num_engines,
        rollout_gpus_per_engine=mode.rollout_gpus_per_engine,
        colocate=True,
        ft_components=("rollout",),
        parallel_args=mode.parallel_args,
    )


def _without_engines(mode: FTTestMode) -> FTTestMode:
    return FTTestMode(
        model_name=mode.model_name,
        model_hf_repo=mode.model_hf_repo,
        megatron_model_type=mode.megatron_model_type,
        num_cells=mode.num_cells,
        train_gpus_per_node=mode.train_gpus_per_node,
        parallel_args=mode.parallel_args,
    )


_FLAGS_A_DEPLOYMENT_MAY_DIFFER_ON: tuple[str, ...] = (
    INFERENCE_CONTROLLER_ADDR_FLAG,
    TRAINER_CONTROLLER_ADDRS_FLAG,
    MOONCAKE_INIT_KWARGS_FLAG,
    scenario.ROLLOUT_NUM_GPUS_FLAG,
)


def _shared_argv(train_args: str) -> list[str]:
    return shared_argv(train_args, differing_flags=_FLAGS_A_DEPLOYMENT_MAY_DIFFER_ON)
