from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class RolloutSamplingMask:
    """One sample's sampling support: the token ids the rollout sampler could
    emit at each response position.

    Stored in CSR form so it stays two flat integer arrays end to end: token
    ``t``'s support is ``ids[offsets[t] : offsets[t + 1]]``. Inputs may be
    tensors or plain integer sequences (the Mooncake object-store codec
    decodes per-sample rows as plain int lists).
    """

    ids: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self):
        object.__setattr__(self, "ids", _to_cpu_integer_tensor(self.ids))
        object.__setattr__(self, "offsets", _to_cpu_integer_tensor(self.offsets))
        if self.offsets.numel() == 0 or self.offsets[0] != 0 or self.offsets[-1] != self.ids.numel():
            raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
        if torch.any(self.offsets[1:] <= self.offsets[:-1]):
            raise ValueError("every response token must have a non-empty sampling support")

    @classmethod
    def from_supports(cls, supports: Sequence[Sequence[int] | torch.Tensor]) -> "RolloutSamplingMask":
        """Build from one support (the allowed token ids) per response token."""
        parts = [_to_cpu_integer_tensor(support) for support in supports]
        lengths = torch.tensor([part.numel() for part in parts], dtype=torch.long)
        ids = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        return cls(ids=ids, offsets=offsets)

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def __getitem__(self, token_index: int) -> torch.Tensor:
        if token_index < 0:
            token_index += len(self)
        if not 0 <= token_index < len(self):
            raise IndexError(f"response token index {token_index} out of range for {len(self)} tokens")
        return self.ids[self.offsets[token_index] : self.offsets[token_index + 1]]


def build_local_sampling_mask(
    logits: torch.Tensor,
    sampling_mask: RolloutSamplingMask,
    response_indices: Sequence[int] | torch.Tensor,
    *,
    response_length: int,
    tp_rank: int,
) -> torch.Tensor:
    """Build the dense local-vocabulary mask consumed by the log-prob primitive."""
    ids = sampling_mask.ids
    offsets = sampling_mask.offsets
    indices = _to_cpu_integer_tensor(response_indices)

    if indices.numel() != logits.size(0):
        raise ValueError(
            f"sampling-mask rows must align with logits: indices={indices.numel()}, logits={logits.size(0)}"
        )
    if len(sampling_mask) != response_length:
        raise ValueError(
            f"sampling mask covers {len(sampling_mask)} response tokens != response length {response_length}"
        )
    if torch.any(indices < 0) or torch.any(indices >= response_length):
        raise ValueError(f"response indices must be in [0, {response_length})")

    local_vocab_size = logits.size(-1)
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    if indices.numel() == 0:
        return mask.view_as(logits)

    indices = indices.to(torch.long)
    lengths = offsets[indices + 1] - offsets[indices]
    # CP response rows form a small number of contiguous runs. Slice those
    # runs on CPU, then expand and TP-filter the CSR data on the GPU.
    run_starts = [0]
    run_starts.extend((torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1).tolist())
    run_starts.append(indices.numel())
    selected_parts = [
        ids[offsets[indices[start]] : offsets[indices[end - 1] + 1]]
        for start, end in zip(run_starts[:-1], run_starts[1:], strict=True)
    ]
    selected_ids = selected_parts[0] if len(selected_parts) == 1 else torch.cat(selected_parts)

    selected_ids = selected_ids.to(logits.device)
    row_indices = torch.repeat_interleave(
        torch.arange(indices.numel(), dtype=torch.long, device=logits.device),
        lengths.to(device=logits.device, dtype=torch.long),
    )
    is_local = (selected_ids >= vocab_start) & (selected_ids < vocab_end)
    flat_local_indices = row_indices[is_local] * local_vocab_size + selected_ids[is_local].to(torch.long) - vocab_start
    mask[flat_local_indices] = True
    return mask.view_as(logits)


def _to_cpu_integer_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    else:
        if isinstance(values, range):
            tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
        else:
            tensor = torch.as_tensor(values, device="cpu")
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
    return tensor
