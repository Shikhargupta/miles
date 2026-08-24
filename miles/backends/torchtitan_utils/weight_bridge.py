"""Stream torchtitan weights to the rollout engines.

The HF naming comes from the model's own ``state_dict_adapter.to_hf`` -- the
same mapping used to load the initial checkpoint, run in reverse. The transport
(IPC buckets to a colocated engine) and the engine handshake are the shared
implementations, so this module is only the weight *production* side.
"""

import logging

import torch

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.fsdp_utils.update_weight_utils import UpdateWeightFromTensor
from miles.backends.training_utils.weight_sync import weight_push_session

logger = logging.getLogger(__name__)


class TitanUpdateWeightFromTensor(UpdateWeightFromTensor):
    """Reuses FSDP's colocated IPC transport; only weight production differs.

    FSDP streams ``model.state_dict()`` under HF names because it trains stock HF
    modeling. torchtitan's parameter names are its own, so each tensor is mapped
    through the state-dict adapter and materialized from its DTensor shards
    before it goes on the wire.
    """

    def __init__(self, args, model: torch.nn.Module, sd_adapter) -> None:
        super().__init__(args, model)
        self._sd_adapter = sd_adapter

    def update_weights(self) -> None:
        self.weight_version += 1
        with weight_push_session(self.args, self.rollout_engines):
            self._stream_weights()

    def _stream_weights(self) -> None:
        bucket: list[tuple[str, torch.Tensor, None]] = []
        bucket_size = 0

        for name, tensor in self._iter_hf_weights():
            size = tensor.numel() * tensor.element_size()
            if bucket and bucket_size + size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                bucket = []
                bucket_size = 0
            bucket.append((name, tensor, None))
            bucket_size += size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)

    def _iter_hf_weights(self):
        """HF-named tensors, materialized one at a time.

        ``to_hf`` runs on the sharded DTensors -- that is how torchtitan's own
        checkpoint path calls it, since DCP consumes the shards directly. Only
        the tensors handed to the transport are materialized, and the bucketing
        in ``_stream_weights`` bounds how many are resident at once. Gathering the
        whole state dict up front instead would put a full unsharded copy of the
        model on every rank, which is fine for a 0.6B and fatal for a 30B.

        Gathering goes through the shared FSDP2 helper, which moves to CUDA
        first: ``full_tensor()`` on a CPU DTensor selects a collective backend
        that is not registered.
        """
        for name, tensor in self._sd_adapter.to_hf(self.model.state_dict()).items():
            yield name, gather_full_param(tensor)
