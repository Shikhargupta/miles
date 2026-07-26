import json
import os
import shlex
from collections.abc import Callable
from functools import partial
from pathlib import Path

from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend
from miles.utils.external_utils.command_utils.common import (
    ExecuteTrainConfig,
    build_runtime_env_vars,
    get_bool_env_var,
    pythonpath_with_sources,
    repo_base_dir,
)
from miles.utils.external_utils.exec_command import exec_command_cpu, exec_command_gpu, exec_command_multi_node


class RayCommandBackend(BaseCommandBackend):
    def execute_train(
        self,
        train_args: str,
        num_gpus_per_node: int,
        megatron_model_type: str | None,
        train_script: str = "train.py",
        before_ray_job_submit: Callable[[], None] | None = None,
        extra_env_vars: dict[str, str] | None = None,
        config: ExecuteTrainConfig | None = None,
        megatron_path: str = "/root/Megatron-LM",
    ) -> None:
        if extra_env_vars is None:
            extra_env_vars = {}
        if config is None:
            config = ExecuteTrainConfig()
        if not os.path.isabs(train_script):
            train_script = f"{repo_base_dir}/{train_script}"
        external_ray = get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY")
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

        train_backend_fsdp = "--train-backend fsdp" in train_args
        assert train_backend_fsdp == (megatron_model_type is None)

        exec_command_cpu(
            "pkill -9 sglang; "
            "sleep 3; "
            f"{'' if external_ray else 'ray stop --force; '}"
            f"{'' if external_ray else 'pkill -9 ray; '}"
            # cannot be run in CI, o/w kill the parent script
            # TODO: do we really need this kill? (or can we instead kill miles)
            # "pkill -9 python; "
            "pkill -9 miles; "
            "sleep 3; "
            f"{'' if external_ray else 'pkill -9 ray; '}"
            # "pkill -9 python; "
            "pkill -9 miles; "
            "pkill -9 redis; "
            "true; "
        )

        if not external_ray:
            exec_command_cpu(
                # will prevent ray from buffering stdout/stderr
                f"export PYTHONUNBUFFERED=1 && "
                f"ray start --head --node-ip-address {master_addr} --num-gpus {num_gpus_per_node} --disable-usage-stats"
            )

        if (f := before_ray_job_submit) is not None:
            f()

        runtime_env_json = json.dumps(
            {
                "env_vars": build_runtime_env_vars(
                    train_backend_fsdp=train_backend_fsdp,
                    master_addr=master_addr,
                    megatron_path=megatron_path,
                    extra_env_vars=extra_env_vars,
                    config=config,
                )
            }
        )

        if get_bool_env_var("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
            cmd_megatron_model_source = (
                f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && '
                if megatron_model_type is not None
                else ""
            )
            exec_command_cpu(
                f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
                f"{cmd_megatron_model_source}"
                f"""ray job submit {'' if 'RAY_ADDRESS' in os.environ else '--address="http://127.0.0.1:8265" '}"""
                f"--runtime-env-json={shlex.quote(runtime_env_json)} "
                f"-- python3 {train_script} "
                f"{'${MODEL_ARGS[@]}' if megatron_model_type is not None else ''} "
                f"{train_args}"
            )

    def convert_checkpoint(
        self,
        model_name,
        megatron_model_type,
        num_gpus_per_node: int,
        multinode: bool = False,
        num_nodes: int | None = None,
        extra_args: str = "",
        dir_dst: str = "/root",
        hf_checkpoint: str | None = None,
        megatron_path: str = "/root/Megatron-LM",
    ) -> None:
        hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"

        # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
        path_dst = f"{dir_dst}/{model_name}_torch_dist"
        tracker = Path(path_dst) / "latest_checkpointed_iteration.txt"
        if tracker.exists() and tracker.read_text().strip() == "release":
            print(f"convert_checkpoint skip {path_dst} since tracker is 'release'")
            return

        multinode_args = ""
        if multinode:
            multinode_args = (
                "--master-addr {{master_addr}} "
                "--master-port 23456 "
                "--nnodes={{nnodes}} "
                "--node-rank {{node_rank}} "
            )

        if multinode:
            fn = partial(exec_command_multi_node, num_nodes=num_nodes)
        else:
            fn = exec_command_gpu
        pythonpath = shlex.quote(pythonpath_with_sources(megatron_path))
        fn(
            f"source {repo_base_dir}/scripts/models/{megatron_model_type}.sh && "
            f"PYTHONPATH={pythonpath} "
            f"torchrun "
            f"--nproc-per-node {num_gpus_per_node} "
            f"{multinode_args}"
            f"{repo_base_dir}/tools/convert_hf_to_torch_dist.py "
            "${MODEL_ARGS[@]} "
            f"--hf-checkpoint {hf_checkpoint} "
            f"--save {path_dst} "
            f"{extra_args}"
        )

    def rsync_simple(self, path_src: str, path_dst: str, num_nodes: int | None = None) -> None:
        exec_command_multi_node(
            f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}", num_nodes=num_nodes
        )

    def hf_download_dataset(self, full_name: str, data_dir: str = "/root/datasets") -> None:
        _, partial_name = full_name.split("/")
        exec_command_cpu(f"hf download --repo-type dataset {full_name} --local-dir {data_dir}/{partial_name}")

    def fp8_cast_bf16(self, path_src, path_dst) -> None:
        sentinel = Path(path_dst) / "model.safetensors.index.json"
        if sentinel.exists():
            print(f"fp8_cast_bf16 skip {path_dst} since {sentinel} exists")
            return

        exec_command_gpu(
            f"python {repo_base_dir}/tools/fp8_cast_bf16.py "
            f"--input-fp8-hf-path {path_src} "
            f"--output-bf16-hf-path {path_dst} "
        )
