import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class RolloutSamplingMask:
    """One sample's sampling mask: for each response position, the token ids
    the rollout sampler could emit.

    Stored in CSR form so it stays two flat integer arrays end to end:
    ``ids`` is ``[total_support_size]`` (all supports concatenated), ``offsets``
    is ``[num_response_tokens + 1]``, and token ``t``'s support is
    ``ids[offsets[t] : offsets[t + 1]]``. That is the shape object-store
    transport needs, so no per-token nesting is rebuilt on the trainer side.
    """

    ids: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self):
        # copy=True takes ownership: a caller mutating its input tensor after
        # construction must not be able to invalidate the checks below.
        object.__setattr__(self, "ids", _to_cpu_integer_tensor(self.ids).to(torch.long, copy=True))
        object.__setattr__(self, "offsets", _to_cpu_integer_tensor(self.offsets).to(torch.long, copy=True))
        if self.offsets.numel() == 0 or self.offsets[0] != 0 or self.offsets[-1] != self.ids.numel():
            raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
        if torch.any(self.offsets[1:] <= self.offsets[:-1]):
            raise ValueError(
                "sampling-mask offsets must be strictly increasing: "
                "every response token needs a non-empty sampling mask"
            )

    @classmethod
    def from_mask_list(cls, mask_list: Sequence[Sequence[int]]) -> "RolloutSamplingMask":
        """Build from one mask (the allowed token ids) per response token.

        Args:
            mask_list: ragged ``[num_response_tokens][mask_size_t]``;
                ``mask_list[t]`` lists the token ids the sampler could emit at
                response position ``t``. SGLang's ``output_token_sampling_mask``
                arrives in this shape.
        """
        ids = _to_cpu_integer_tensor([token_id for mask in mask_list for token_id in mask])
        lengths = torch.tensor([len(mask) for mask in mask_list], dtype=torch.long)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        return cls(ids=ids, offsets=offsets)

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def __getitem__(self, token_index: int) -> torch.Tensor:
        """Mask ids at one response position, shape ``[mask_size_t]``."""
        token_index = operator.index(token_index)
        if not 0 <= token_index < len(self):
            raise IndexError(f"response token index {token_index} out of range for {len(self)} tokens")
        return self.ids[self.offsets[token_index] : self.offsets[token_index + 1]]

    def select_masks(self, token_indices: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened masks for the given response positions.

        Args:
            token_indices: ``[num_selected]`` response positions to read.

        Returns:
            ``(ids, lengths)`` where ``ids`` is ``[sum(lengths)]``, the selected
            masks concatenated in ``token_indices`` order, and ``lengths`` is
            ``[num_selected]``, each position's mask size.

        Consecutive token indices share one CSR run, so whole runs are sliced
        at once: O(#runs) slices instead of one per token.
        """
        indices = _to_cpu_integer_tensor(token_indices).to(torch.long)
        if torch.any(indices < 0) or torch.any(indices >= len(self)):
            raise ValueError(f"response indices must be in [0, {len(self)})")
        lengths = self.offsets[indices + 1] - self.offsets[indices]
        if indices.numel() == 0:
            return self.ids.new_empty(0), lengths
        run_starts = [0]
        run_starts.extend((torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1).tolist())
        run_starts.append(indices.numel())
        parts = [
            self.ids[self.offsets[indices[start]] : self.offsets[indices[end - 1] + 1]]
            for start, end in zip(run_starts[:-1], run_starts[1:], strict=True)
        ]
        # cat always copies, so the result never aliases the frozen storage.
        return torch.cat(parts), lengths


def _to_cpu_integer_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    elif len(values) == 0:
        tensor = torch.empty(0, dtype=torch.long, device="cpu")
    elif isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
    return tensor
