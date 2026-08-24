"""Stream torchtitan weights to the rollout engines.

The HF naming comes from the model's own ``state_dict_adapter.to_hf`` -- the
same mapping used to load the initial checkpoint, run in reverse -- which the
engine exposes as ``hf_weights()``. The transport (IPC buckets to a colocated
engine) and the engine handshake are the shared implementations, so this module
is only the weight *production* side.
"""

import logging

import torch

from miles.backends.fsdp_utils.update_weight_utils import UpdateWeightFromTensor
from miles.backends.training_utils.weight_sync import weight_push_session

logger = logging.getLogger(__name__)


class TitanUpdateWeightFromTensor(UpdateWeightFromTensor):
    """Reuses FSDP's colocated IPC transport; only weight production differs.

    FSDP streams ``model.state_dict()`` under HF names because it trains stock
    HF modeling. torchtitan's parameter names are its own, so the engine maps
    each tensor through its state-dict adapter and materializes it from its
    DTensor shards before it goes on the wire. Only the tensors handed to the
    transport are materialized, and the bucketing bounds how many are resident
    at once; gathering the whole state dict up front would put a full unsharded
    copy of the model on every rank, fine for a 0.6B and fatal for a 30B.
    """

    def __init__(self, args, engine) -> None:
        super().__init__(args, engine.model_parts[0])
        self._engine = engine

    def update_weights(self) -> None:
        self.weight_version += 1
        with weight_push_session(self.args, self.rollout_engines):
            self._stream_weights()

    def _stream_weights(self) -> None:
        bucket: list[tuple[str, torch.Tensor, None]] = []
        bucket_size = 0

        for name, tensor in self._engine.hf_weights():
            size = tensor.numel() * tensor.element_size()
            if bucket and bucket_size + size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                bucket = []
                bucket_size = 0
            bucket.append((name, tensor, None))
            bucket_size += size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)
