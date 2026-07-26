from abc import ABC, abstractmethod
from collections.abc import Callable

from miles.utils.external_utils.command_utils.common import ExecuteTrainConfig


class BaseCommandBackend(ABC):
    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
    def rsync_simple(self, path_src: str, path_dst: str, num_nodes: int | None = None) -> None: ...

    @abstractmethod
    def hf_download_dataset(self, full_name: str, data_dir: str = "/root/datasets") -> None: ...

    @abstractmethod
    def fp8_cast_bf16(self, path_src, path_dst) -> None: ...
