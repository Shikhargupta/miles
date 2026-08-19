import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import (
    LAYOUT,
    engine,
    router,
    session_server,
    trainer,
)

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.helm_backend.launcher.values import pool_entry
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import BaseWorkerSpec

STAMP = "2026-08-12T09:00:00+00:00"

RESTARTED_LAYOUT = LAYOUT.model_copy(
    update=dict(restart_at=STAMP, stamped_components=frozenset({ROLLOUT_EXECUTOR_POOL_ID}))
)

PREPARE_CMD = "mkdir -p /scratch/dataset && rsync -a /cluster-storage/dataset/ /scratch/dataset"

PREPARE_LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release="r",
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
    prepare_cmd={"trainer": PREPARE_CMD},
)


def _prepared(section: str, spec: BaseWorkerSpec) -> list[str]:
    return build_values([spec], PREPARE_LAYOUT).as_values()["run"][section][0]["command"]


class TestBuildEntry:
    def test_refuses_to_build_an_entry_for_a_spec_that_wants_no_cell(self):
        """Every entry renders at least one pod, so a zero-cell spec reaching conversion is a silent deploy."""
        with pytest.raises(AssertionError, match="cells"):
            pool_entry.build_entry(session_server(num_cells=0), plan=LAYOUT, addresses={})

    def test_replaces_only_the_node_rank_with_the_kubelet_placeholder(self):
        """Every pod of a group shares one command, so the rank must be the one part left to kubelet."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--node-rank") + 1] == pool_entry._WORKER_INDEX_PLACEHOLDER
        assert command[command.index("--base-gpu-id") + 1] == "0"

    def test_points_a_master_port_at_the_group_leader(self):
        """A rank cannot know its leader's address until it is scheduled, but kubelet does."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--dist-init-addr") + 1] == f"{pool_entry._LEADER_ADDRESS_PLACEHOLDER}:9000"

    def test_allows_a_command_that_never_mentions_its_rank(self):
        """Some engines do not take one; what matters is that no sentinel survives into the command."""
        spec = engine().model_copy(update={"launch_command": lambda ctx: "python -m sglang.launch_server"})

        command = build_values([spec], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert str(pool_entry._WORKER_INDEX_SENTINEL) not in " ".join(command)


class TestPrepareCmd:
    def test_a_trainer_runs_the_command_before_it_starts(self):
        """Every trainer rank reads the dataset every step, and the shared filesystem cannot serve that."""
        command = _prepared("trainerEngines", trainer())

        assert command[:2] == ["bash", "-c"]
        assert command[2].startswith(f"{PREPARE_CMD} && exec ")

    def test_an_engine_is_launched_without_the_command(self):
        """An sglang engine loads weights the trainer sends it, so the copy would move gigabytes nothing reads."""
        command = _prepared("inferenceEngines", engine())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_static_worker_is_launched_without_the_command(self):
        """A router holds no data of its own, and the copy would only delay the run's first request."""
        command = _prepared("staticWorkers", router())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_trainer_keeps_the_command_it_would_have_run(self):
        """The preparation prefixes the launch; a rewritten command would start the wrong worker."""
        command = _prepared("trainerEngines", trainer())

        assert command[2].endswith(" ".join(["--foo", "bar"]))

    def test_quotes_the_command_it_wraps(self):
        """An argument carrying json or spaces would otherwise be re-split by the shell that runs the pair."""
        spec = trainer(num_cells=1, gpus_per_cell=8)
        layout = PREPARE_LAYOUT.model_copy(update={"worker_argv": ["--kwargs", '{"a": 1}']})

        command = build_values([spec], layout).as_values()["run"]["trainerEngines"][0]["command"]

        assert command[2].endswith("""--kwargs \'{"a": 1}\'""")

    def test_refuses_a_trainer_whose_pods_can_share_a_node(self):
        """Two pods of one node would run the copy against the same node-local path at the same time."""
        with pytest.raises(AssertionError, match="can land on one node"):
            build_values([trainer(num_cells=1, gpus_per_cell=4)], PREPARE_LAYOUT).as_values()

    def test_accepts_a_trainer_that_takes_whole_nodes(self):
        """A pod holding every gpu of its node is the only pod there, so the copy cannot race itself."""
        command = _prepared("trainerEngines", trainer(num_cells=1, gpus_per_cell=8))

        assert command[:2] == ["bash", "-c"]

    def test_a_run_that_prepares_nothing_leaves_every_command_alone(self):
        """Most runs read from the shared filesystem directly, and a bash wrapper would hide their exit codes."""
        command = build_values([trainer()], LAYOUT).as_values()["run"]["trainerEngines"][0]["command"]

        assert command[:2] != ["bash", "-c"]


class TestTheRestartStamp:
    @staticmethod
    def _entry(spec: BaseWorkerSpec, *, plan: LaunchPlan) -> dict:
        return build_values([spec], plan).as_values()["run"]["staticWorkers"][0]

    @staticmethod
    def _executor() -> BaseWorkerSpec:
        return router().model_copy(update={"name": ROLLOUT_EXECUTOR_POOL_ID})

    def test_a_pool_the_launch_replaces_carries_the_stamp(self):
        """This is the only path from the plan to the annotation whose change rolls the executor pod."""
        assert self._entry(self._executor(), plan=RESTARTED_LAYOUT)["restartAt"] == STAMP

    def test_a_pool_the_launch_does_not_replace_carries_none(self):
        """Stamping any other pool would roll pods this launch promises to keep alive."""
        assert "restartAt" not in self._entry(router(), plan=RESTARTED_LAYOUT)

    def test_an_ordinary_launch_stamps_no_pool_at_all(self):
        """A stamp appearing without a restart would roll the executor of every run that relaunches."""
        assert "restartAt" not in self._entry(self._executor(), plan=LAYOUT)

    def test_stamping_a_pool_whose_template_renders_no_annotation_is_refused(self):
        """The engine template carries no annotation, so a stamp there rolls nothing while the launch believes it did."""
        plan = LAYOUT.model_copy(update=dict(restart_at=STAMP, stamped_components=frozenset({"inference-engine-0-0"})))

        with pytest.raises(AssertionError, match="renders a restart stamp"):
            build_values([engine()], plan).as_values()

    def test_stamping_a_trainer_pool_is_refused(self):
        """A trainer pool a hot restart promises to keep alive must never be handed a stamp at all."""
        plan = LAYOUT.model_copy(update=dict(restart_at=STAMP, stamped_components=frozenset({"trainer-engine-actor"})))

        with pytest.raises(AssertionError, match="renders a restart stamp"):
            build_values([trainer()], plan).as_values()


COLOCATE_LAYOUT = LAYOUT.model_copy(update={"colocate": True})


def _engine_command(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> list[str]:
    return build_values(specs, plan).as_values()["run"]["inferenceEngines"][0]["command"]


def _base_gpu_id_argument(command: list[str]) -> str:
    return command[command.index("--base-gpu-id") + 1]


class TestBaseGpuIdOfASubNodeEngine:
    def test_leaves_a_whole_node_engine_to_be_handed_its_cards_from_zero(self):
        """A pod holding every card of its node is given them as devices 0..n-1, so nothing has to be told."""
        command = _engine_command(
            [engine(num_cells=2, gpus_per_engine=8), trainer(num_cells=2, gpus_per_cell=8)], COLOCATE_LAYOUT
        )

        assert _base_gpu_id_argument(command) == "0"

    def test_leaves_a_disaggregated_engine_alone_even_when_it_is_narrower_than_a_node(self):
        """A pool past the trainer's gpus owns its nodes, so its cards are its own and start at zero."""
        specs = [
            engine(num_cells=1, gpus_per_engine=4, name="inference-engine-0-0", gpu_offset=0),
            engine(num_cells=1, gpus_per_engine=4, name="inference-engine-0-1", gpu_offset=16),
            trainer(num_cells=2, gpus_per_cell=8),
        ]

        entries = build_values(specs, COLOCATE_LAYOUT).as_values()["run"]["inferenceEngines"]
        past_the_trainer = next(entry for entry in entries if entry["poolId"] == "inference-engine-0-1")

        assert _base_gpu_id_argument(past_the_trainer["command"]) == "0"

    def test_asks_the_kubelet_for_the_card_an_engine_sharing_a_trainer_node_was_given(self):
        """Which card a shared node hands this pod is decided when the pairing seats it, not when helm renders."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert _base_gpu_id_argument(command) == "$(MILES_BASE_GPU_ID)"

    def test_substitutes_the_card_as_a_whole_argument_of_its_own(self):
        """The kubelet expands a whole argument at a time, so a sentinel welded into one never resolves."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert str(pool_entry._BASE_GPU_ID_SENTINEL) not in " ".join(command)

    def test_wraps_the_command_in_no_shell_of_its_own(self):
        """The value arrives already computed, so nothing in the pod has to redo the pairing's arithmetic."""
        specs = [engine(num_cells=2, gpus_per_engine=4), trainer(num_cells=1, gpus_per_cell=8)]

        command = _engine_command(specs, COLOCATE_LAYOUT)

        assert command[:2] != ["bash", "-c"]

    def test_refuses_a_spec_that_builds_the_card_into_a_larger_argument(self):
        """Kubelet substitutes whole arguments, so a spelling like --gpus=N would reach sglang unexpanded."""
        spec = engine(num_cells=2, gpus_per_engine=4).model_copy(
            update={"launch_command": lambda ctx: f"python -m sglang.launch_server --base-gpu-id={ctx.gpu_ids[0]}"}
        )

        with pytest.raises(AssertionError, match="base gpu id"):
            build_values([spec, trainer(num_cells=1, gpus_per_cell=8)], COLOCATE_LAYOUT).as_values()


class TestTheAccountAPoolRunsUnder:
    def test_a_pool_that_observes_the_platform_gets_the_account_that_may_read_it(self):
        """Only these workers reconcile against pods, and the namespace default cannot list one."""
        spec = session_server(num_cells=1).model_copy(update={"needs_platform_read_permission": True})

        entry = build_values([spec], LAYOUT).as_values()["run"]["staticWorkers"][0]

        assert entry["serviceAccountName"] == "r-miles-run-orchestrator"

    def test_every_other_pool_stays_on_the_namespace_default(self):
        """An engine talks to no api server, and an account it never needs is one it could misuse."""
        entry = build_values([session_server(num_cells=1)], LAYOUT).as_values()["run"]["staticWorkers"][0]

        assert "serviceAccountName" not in entry

    def test_refuses_a_pool_whose_template_renders_no_account_at_all(self):
        """The engine template ignores the key, so the pod would run on the default and 403 far from here."""
        spec = engine(num_cells=1, gpus_per_engine=8).model_copy(update={"needs_platform_read_permission": True})

        with pytest.raises(AssertionError, match="renders a service account"):
            build_values([spec], LAYOUT).as_values()
