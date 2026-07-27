import json
import shlex

import pytest
from tests.fast.utils.command_recorder import record_commands

import miles.utils.external_utils.command_utils as command_utils
import miles.utils.external_utils.command_utils.common as common


@pytest.fixture
def commands(monkeypatch):
    recorded = record_commands(monkeypatch)
    monkeypatch.setattr(common, "check_has_nvlink", lambda: False)
    for name in ("MILES_SCRIPT_EXTERNAL_RAY", "RAY_ADDRESS", "NCCL_NVLS_ENABLE", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    return recorded


def _runtime_env(submit_command):
    arg = next(arg for arg in shlex.split(submit_command) if arg.startswith("--runtime-env-json="))
    return json.loads(arg.split("=", 1)[1])["env_vars"]


class TestConvertCheckpoint:
    def test_defaults_the_hf_checkpoint_to_the_model_name(self, commands, tmp_path):
        """Callers that only pass a model name get /root/models/<model_name> as the source."""
        command_utils.convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert "--hf-checkpoint /root/models/Qwen3-4B " in commands[0]
        assert f"--save {tmp_path}/Qwen3-4B_torch_dist " in commands[0]

    def test_skips_an_already_released_destination(self, commands, tmp_path):
        """A tracker file reading 'release' means the conversion already finished."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("release\n")

        command_utils.convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert commands == []

    def test_reruns_when_the_tracker_holds_an_iteration(self, commands, tmp_path):
        """Only the literal 'release' marker counts as done; an iteration number does not."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("42")

        command_utils.convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert len(commands) == 1

    def test_multinode_uses_torchrun_rendezvous_placeholders(self, commands, tmp_path):
        """Multi-node conversion must template the placeholders exec_command_multi_node substitutes."""
        command_utils.convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=8,
            multinode=True,
            num_nodes=2,
            dir_dst=str(tmp_path),
            extra_args="--extra 1",
        )

        assert "--master-addr {{master_addr}}" in commands[0]
        assert "--nnodes={{nnodes}}" in commands[0]
        assert "--node-rank {{node_rank}}" in commands[0]
        assert commands[0].endswith("--extra 1")

    def test_single_node_omits_the_rendezvous_placeholders(self, commands, tmp_path):
        """A single-node conversion has nothing to rendezvous with."""
        command_utils.convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert "--master-addr" not in commands[0]


class TestRsyncSimple:
    def test_creates_the_destination_before_copying(self, commands):
        """rsync fails on a missing destination, so the mkdir has to precede it."""
        command_utils.rsync_simple("/src", "/dst")

        assert commands == ["[multi_node num_nodes=None] mkdir -p /dst && rsync -a --info=progress2 /src/ /dst"]


class TestHfDownloadDataset:
    def test_strips_the_namespace_from_the_local_dir(self, commands):
        """The local directory is named after the dataset, not after owner/dataset."""
        command_utils.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir="/data")

        assert commands == ["hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /data/dapo-math-17k"]


class TestFp8CastBf16:
    def test_skips_when_the_output_index_already_exists(self, commands, tmp_path):
        """A safetensors index in the destination means the cast already ran."""
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        command_utils.fp8_cast_bf16("/src", str(tmp_path))

        assert commands == []

    def test_runs_when_the_output_is_absent(self, commands, tmp_path):
        """Without the index file the cast must actually be invoked."""
        command_utils.fp8_cast_bf16("/src", str(tmp_path))

        assert "--input-fp8-hf-path /src " in commands[0]
        assert f"--output-bf16-hf-path {tmp_path} " in commands[0]


class TestExecuteTrain:
    def test_rejects_a_backend_that_disagrees_with_the_model_type(self, commands):
        """FSDP runs have no megatron model type, and megatron runs must have one."""
        with pytest.raises(AssertionError):
            command_utils.execute_train(
                train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type="qwen"
            )

    def test_starts_a_local_ray_cluster_by_default(self, commands):
        """Without MILES_SCRIPT_EXTERNAL_RAY the launcher owns the ray cluster lifecycle."""
        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "ray stop --force; " in commands[0]
        assert "ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats" in commands[1]

    def test_leaves_an_external_ray_cluster_alone(self, commands, monkeypatch):
        """With an external cluster we must neither stop nor start ray."""
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert not any("ray stop" in command or "ray start" in command for command in commands)

    def test_runs_the_callback_before_submitting(self, commands):
        """before_ray_job_submit exists to prepare state the job will read."""
        command_utils.execute_train(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            before_ray_job_submit=lambda: commands.append("CALLBACK"),
        )

        assert commands.index("CALLBACK") < len(commands) - 1
        assert "ray job submit" in commands[-1]

    def test_can_skip_the_ray_job_submit(self, commands, monkeypatch):
        """Preparation-only runs disable the submit but still clean up and start ray."""
        monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "0")

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert not any("ray job submit" in command for command in commands)

    def test_sources_the_model_config_and_expands_model_args(self, commands):
        """The megatron model type is turned into a `source` plus a ${MODEL_ARGS[@]} expansion."""
        command_utils.execute_train(train_args="--x 1", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        submit = commands[-1]
        assert f'source "{command_utils.repo_base_dir}/scripts/models/qwen3-4B.sh" && ' in submit
        assert "${MODEL_ARGS[@]}" in submit
        assert submit.endswith("--x 1")

    def test_omits_the_model_source_for_fsdp(self, commands):
        """FSDP has no megatron model config to source."""
        command_utils.execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

        assert "scripts/models/" not in commands[-1]
        assert "${MODEL_ARGS[@]}" not in commands[-1]

    def test_drops_cuda_device_max_connections_for_fsdp(self, commands):
        """Pinning it to 1 breaks computation/communication overlap on FSDP."""
        command_utils.execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

        assert "CUDA_DEVICE_MAX_CONNECTIONS" not in _runtime_env(commands[-1])

    def test_pins_cuda_device_max_connections_for_megatron(self, commands):
        """Megatron requires the serialized copy engine ordering."""
        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(commands[-1])["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"

    def test_derives_nvls_from_nvlink_detection(self, commands, monkeypatch):
        """NCCL_NVLS_ENABLE follows the detected topology when it is not preset."""
        monkeypatch.setattr(common, "check_has_nvlink", lambda: True)

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(commands[-1])["NCCL_NVLS_ENABLE"] == "1"

    def test_lets_the_environment_override_nvls(self, commands, monkeypatch):
        """An explicit NCCL_NVLS_ENABLE wins over topology detection."""
        monkeypatch.setattr(common, "check_has_nvlink", lambda: True)
        monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(commands[-1])["NCCL_NVLS_ENABLE"] == "0"

    def test_forwards_selected_nccl_variables_only_when_present(self, commands, monkeypatch):
        """Optional debug knobs are passed through, and absent ones must not appear as empty strings."""
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        monkeypatch.delenv("NCCL_DEBUG", raising=False)

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(commands[-1])
        assert runtime_env["NCCL_SOCKET_IFNAME"] == "eth0"
        assert "NCCL_DEBUG" not in runtime_env

    def test_bypasses_the_proxy_for_the_master_node(self, commands, monkeypatch):
        """Routing intra-cluster traffic through a proxy hangs the job."""
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(commands[-1])
        assert runtime_env["no_proxy"] == "127.0.0.1,10.0.0.1"
        assert runtime_env["MASTER_ADDR"] == "10.0.0.1"

    def test_enables_cuda_core_dumps_on_request(self, commands):
        """The core dump knobs only appear when the config asks for them."""
        config = command_utils.ExecuteTrainConfig(cuda_core_dump=True, output_dir="/out")

        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B", config=config)

        runtime_env = _runtime_env(commands[-1])
        assert runtime_env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
        assert runtime_env["CUDA_COREDUMP_FILE"] == "/out/cuda_coredump_%h.%p.%t"

    def test_omits_cuda_core_dumps_by_default(self, commands):
        """Core dumps are expensive, so they must stay off unless asked for."""
        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "CUDA_ENABLE_COREDUMP_ON_EXCEPTION" not in _runtime_env(commands[-1])

    def test_lets_config_extra_env_vars_win_over_the_argument(self, commands):
        """The CLI-supplied overrides are applied last so an operator can always override a script."""
        config = command_utils.ExecuteTrainConfig(extra_env_vars="MY_VAR=from_config")

        command_utils.execute_train(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            extra_env_vars={"MY_VAR": "from_argument", "OTHER": "kept"},
            config=config,
        )

        runtime_env = _runtime_env(commands[-1])
        assert runtime_env["MY_VAR"] == "from_config"
        assert runtime_env["OTHER"] == "kept"

    def test_addresses_the_local_dashboard_unless_ray_address_is_set(self, commands, monkeypatch):
        """RAY_ADDRESS already tells the ray CLI where to go; passing --address too would conflict."""
        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert '--address="http://127.0.0.1:8265"' in commands[-1]

        monkeypatch.setenv("RAY_ADDRESS", "http://10.0.0.1:8265")
        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert "--address=" not in commands[-1]

    def test_resolves_a_relative_train_script_against_the_repo(self, commands):
        """Launchers pass train.py, which only makes sense relative to the checkout."""
        command_utils.execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert f"-- python3 {command_utils.repo_base_dir}/train.py " in commands[-1]

    def test_keeps_an_absolute_train_script(self, commands):
        """An absolute path is already unambiguous and must not be rewritten."""
        command_utils.execute_train(
            train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B", train_script="/opt/train.py"
        )

        assert "-- python3 /opt/train.py " in commands[-1]


class TestParseExtraEnvVars:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"A": "1", "B": "2"}', {"A": "1", "B": "2"}),
            ("A=1 B=2", {"A": "1", "B": "2"}),
            ("", {}),
            ("   ", {}),
        ],
    )
    def test_accepts_json_and_shell_style(self, text, expected):
        """Operators pass either a JSON object or plain KEY=VALUE pairs."""
        assert command_utils.parse_extra_env_vars(text) == expected


class TestCheckHasNvlink:
    def test_reports_true_when_links_are_counted(self, monkeypatch):
        """A non-zero NVLink count from nvidia-smi means NVLink is present."""
        monkeypatch.setattr(common, "exec_command_gpu", lambda cmd, capture_output=False: "4\n")

        assert command_utils.check_has_nvlink() is True

    def test_reports_false_without_links(self, monkeypatch):
        """Zero counted links means no NVLink."""
        monkeypatch.setattr(common, "exec_command_gpu", lambda cmd, capture_output=False: "0\n")

        assert command_utils.check_has_nvlink() is False


class TestGetDefaultWandbArgs:
    def test_is_empty_without_an_api_key(self, monkeypatch):
        """Unconfigured wandb must not inject half-populated flags."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        assert command_utils.get_default_wandb_args("tests/fast/utils/test_thing.py") == ""

    def test_names_the_project_after_the_test_file(self, monkeypatch):
        """The project name is how runs are found later, so it tracks the test file."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.delenv("GITHUB_COMMIT_NAME", raising=False)

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/test_qwen3_4b.py", run_id="RUNID")

        assert "--wandb-project miles-test_qwen3_4b " in args
        assert "--wandb-group RUNID " in args
        assert "--wandb-key 'secret' " in args

    def test_qualifies_a_short_test_name_with_its_directory(self, monkeypatch):
        """Short stems like 'run.py' are ambiguous on their own."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/run.py", run_id="RUNID")

        assert "--wandb-project miles-megatron_run " in args

    def test_decorates_the_group_with_commit_and_prefix(self, monkeypatch):
        """CI runs need the commit in the group name, and callers may add their own prefix."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "abc123")

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/test_qwen3_4b.py", "myprefix", run_id="RUNID")

        assert "--wandb-group myprefix_RUNID_abc123 " in args


class TestCreateRunId:
    def test_is_a_timestamp_with_a_random_suffix(self):
        """Concurrent runs on one machine must not collide on the run id."""
        date_part, time_part, random_part = command_utils.create_run_id().split("-")

        assert len(date_part) == 6 and date_part.isdigit()
        assert len(time_part) == 6 and time_part.isdigit()
        assert len(random_part) == 3 and random_part.isdigit()


class TestGetBoolEnvVar:
    @pytest.mark.parametrize(
        "value, expected",
        [("true", True), ("TRUE", True), ("1", True), ("false", False), ("0", False), ("maybe", False)],
    )
    def test_understands_the_usual_spellings(self, monkeypatch, value, expected):
        """Anything not recognizably truthy is treated as false rather than raising."""
        monkeypatch.setenv("SOME_FLAG", value)

        assert command_utils.get_bool_env_var("SOME_FLAG") is expected

    def test_falls_back_to_the_supplied_default(self, monkeypatch):
        """An unset variable takes the default, which is itself parsed as a string."""
        monkeypatch.delenv("SOME_FLAG", raising=False)

        assert command_utils.get_bool_env_var("SOME_FLAG") is False
        assert command_utils.get_bool_env_var("SOME_FLAG", "1") is True


class TestGetEnvEnableInfiniteRun:
    def test_defaults_to_off(self, monkeypatch):
        """Infinite runs must be opt-in; a stuck CI job is expensive."""
        monkeypatch.delenv("MILES_TEST_ENABLE_INFINITE_RUN", raising=False)
        assert command_utils.get_env_enable_infinite_run() is False

        monkeypatch.setenv("MILES_TEST_ENABLE_INFINITE_RUN", "true")
        assert command_utils.get_env_enable_infinite_run() is True


class TestSaveToTempFile:
    def test_writes_the_content_and_returns_a_unique_path(self):
        """Config text handed to a subprocess has to exist on disk under a collision-free name."""
        first = command_utils.save_to_temp_file("hello: world", "yaml")
        second = command_utils.save_to_temp_file("hello: world", "yaml")

        assert first != second
        assert first.endswith(".yaml")
        with open(first) as f:
            assert f.read() == "hello: world"


class TestHardwareTables:
    def test_every_hardware_with_a_generation_also_has_a_gpu_count(self):
        """Every launcher reads the GPU count, while only some read the generation."""
        assert command_utils.GENERATION_HARDWARE.keys() <= command_utils.NUM_GPUS_OF_HARDWARE.keys()
