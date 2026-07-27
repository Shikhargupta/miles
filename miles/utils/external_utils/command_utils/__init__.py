from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend
from miles.utils.external_utils.command_utils.common import (
    GENERATION_HARDWARE,
    NUM_GPUS_OF_HARDWARE,
    ExecuteTrainConfig,
    check_has_nvlink,
    create_run_id,
    get_bool_env_var,
    get_default_wandb_args,
    get_env_enable_infinite_run,
    parse_extra_env_vars,
    pythonpath_with_sources,
    repo_base_dir,
    save_to_temp_file,
)
from miles.utils.external_utils.command_utils.ray_backend import RayCommandBackend
from miles.utils.external_utils.exec_command import exec_command_cpu, exec_command_gpu, exec_command_multi_node
from miles.utils.typer_utils import dataclass_cli

__all__ = [
    "BaseCommandBackend",
    "ExecuteTrainConfig",
    "GENERATION_HARDWARE",
    "NUM_GPUS_OF_HARDWARE",
    "RayCommandBackend",
    "check_has_nvlink",
    "convert_checkpoint",
    "create_run_id",
    "dataclass_cli",
    "exec_command_cpu",
    "exec_command_gpu",
    "exec_command_multi_node",
    "execute_train",
    "fp8_cast_bf16",
    "get_bool_env_var",
    "get_default_wandb_args",
    "get_env_enable_infinite_run",
    "hf_download_dataset",
    "parse_extra_env_vars",
    "pythonpath_with_sources",
    "repo_base_dir",
    "rsync_simple",
    "save_to_temp_file",
]

backend = RayCommandBackend()

convert_checkpoint = backend.convert_checkpoint
execute_train = backend.execute_train
fp8_cast_bf16 = backend.fp8_cast_bf16
hf_download_dataset = backend.hf_download_dataset
rsync_simple = backend.rsync_simple
