"""Stream torchtitan weights to the rollout engines.

The HF naming comes from the model's own ``state_dict_adapter.to_hf`` -- the
same mapping the trainer's checkpointer used to load the initial checkpoint,
run in reverse -- exposed by the trainer as ``hf_weights()``. The transport
(IPC buckets to a colocated engine) and the engine handshake are the shared
implementations, so this module is only the weight *production* side.
"""

import logging

import torch

from miles.backends.fsdp_utils.update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor
from miles.backends.training_utils.weight_sync import weight_push_session

logger = logging.getLogger(__name__)

# Names sglang fuses into a single parameter by caching the halves and writing
# only once both have arrived. The cache lives inside one
# update_weights_from_tensor call, so a bucket boundary between them leaves the
# fused parameter holding stale values -- silently, since nothing errors. The
# halves must therefore share a bucket. Only DeepSeek's MLA down-projections
# work this way: sglang's other fusions (gate_proj/up_proj -> gate_up_proj) go
# through a weight loader that takes a shard id and writes its slice directly,
# so their halves are independent.
_FUSED_SIBLINGS = (("q_a_proj", "kv_a_proj_with_mqa"),)


def _fused_group_key(name: str) -> str | None:
    """The key naming the fused parameter this tensor is half of, or None."""
    for first, second in _FUSED_SIBLINGS:
        for token in (first, second):
            if token in name:
                return name.replace(token, "<fused>")
    return None


class _TitanWeightProducer:
    """Weight production for torchtitan, independent of how they are shipped.

    FSDP streams ``model.state_dict()`` under HF names because it trains stock
    HF modeling. torchtitan's parameter names are its own, so the trainer maps
    each tensor through its state-dict adapter and materializes it from its
    DTensor shards (pp-broadcast included) before it goes on the wire. Only
    the tensors handed to the transport are materialized, and the bucketing
    bounds how many are resident at once; gathering the whole state dict up
    front would put a full unsharded copy of the model on every rank, fine for
    a 0.6B and fatal for a 30B.

    Mixed in ahead of a transport class, so ``_stream_weights`` is titan's
    while ``connect_rollout_engines`` and ``update_bucket_weights`` stay the
    shared implementations.
    """

    def __init__(self, args, trainer) -> None:
        super().__init__(args, trainer.model_parts[0])
        self._trainer = trainer

    def update_weights(self) -> None:
        self.weight_version += 1
        with weight_push_session(self.args, self.rollout_engines):
            self._stream_weights()

    def _stream_weights(self) -> None:
        bucket: list[tuple[str, torch.Tensor, None]] = []
        bucket_size = 0
        pending: dict[str, list[tuple[str, torch.Tensor]]] = {}

        for name, tensor in self._trainer.hf_weights():
            group_key = _fused_group_key(name)
            if group_key is not None:
                # Hold the first half until its sibling shows up, then place
                # both in one bucket. The stream is name-sorted, so siblings
                # are not adjacent and a boundary between them is likely.
                half = pending.setdefault(group_key, [])
                half.append((name, tensor))
                if len(half) < 2:
                    continue
                group = pending.pop(group_key)
            else:
                group = [(name, tensor)]

            size = sum(t.numel() * t.element_size() for _, t in group)
            if bucket and bucket_size + size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                bucket = []
                bucket_size = 0
            bucket.extend((n, t, None) for n, t in group)
            bucket_size += size

        if pending:
            raise RuntimeError(
                f"fused-parameter halves never completed: {sorted(pending)} -- sglang would leave "
                "those parameters stale, so fail rather than push a partial update"
            )
        if bucket:
            self.wait_and_update_bucket_weights(bucket)


class TitanUpdateWeightFromTensor(_TitanWeightProducer, UpdateWeightFromTensor):
    """Colocated: the engine shares the rank's device, so buckets go over IPC."""


class TitanUpdateWeightFromDistributed(_TitanWeightProducer, UpdateWeightFromDistributed):
    """Disaggregated: the engines are on their own GPUs, so rank 0 broadcasts
    each bucket over a temporary NCCL group.

    Every rank still walks the whole weight stream: producing a tensor takes
    collectives (the DTensor gather, and the owner broadcast that completes the
    stream under PP or EP), so a rank that skipped ahead would hang the others.
    Only the final push is rank 0's.
    """
