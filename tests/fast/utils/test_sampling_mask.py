import pytest
import torch

from miles.utils.sampling_mask import RolloutSamplingMask


def test_rollout_sampling_mask_reads_one_mask_per_token():
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7]])

    assert len(mask) == 3
    assert mask.ids.tolist() == [1, 3, 4, 2, 5, 6, 7]
    assert mask.offsets.tolist() == [0, 2, 4, 7]
    assert mask[0].tolist() == [1, 3]
    assert mask[2].tolist() == [5, 6, 7]
    with pytest.raises(IndexError, match="response token index -1 out of range"):
        mask[-1]


def test_rollout_sampling_mask_requires_non_empty_mask():
    with pytest.raises(ValueError, match="every response token needs a non-empty sampling mask"):
        RolloutSamplingMask.from_mask_list([[], [1]])


def test_rollout_sampling_mask_owns_long_storage():
    ids = torch.tensor([5, 7], dtype=torch.int16)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    mask = RolloutSamplingMask(ids=ids, offsets=offsets)

    ids[0] = 99
    offsets[1] = 99

    assert mask.ids.dtype == torch.long and mask.offsets.dtype == torch.long
    assert mask.ids.tolist() == [5, 7]
    assert mask.offsets.tolist() == [0, 1, 2]


def test_select_masks_returns_tensors_that_do_not_alias_the_mask():
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2]])

    ids, _ = mask.select_masks([0, 1])
    ids[0] = 99

    assert mask.ids.tolist() == [1, 3, 4, 2]


def test_rollout_sampling_mask_validates_csr_offsets():
    with pytest.raises(ValueError, match="offsets must start at zero and end at the flattened id count"):
        RolloutSamplingMask(ids=torch.tensor([0, 1]), offsets=torch.tensor([0, 1]))


def test_select_masks_slices_contiguous_runs_and_reports_lengths():
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7], [8]])

    ids, lengths = mask.select_masks(torch.tensor([0, 1, 3]))

    assert ids.tolist() == [1, 3, 4, 2, 8]
    assert lengths.tolist() == [2, 2, 1]


def test_select_masks_handles_no_tokens():
    mask = RolloutSamplingMask.from_mask_list([[1]])

    ids, lengths = mask.select_masks(torch.empty(0, dtype=torch.long))

    assert ids.numel() == 0
    assert lengths.numel() == 0


def test_select_masks_rejects_out_of_range_position():
    mask = RolloutSamplingMask.from_mask_list([[0]])

    with pytest.raises(ValueError, match=r"response indices must be in \[0, 1\)"):
        mask.select_masks([1])
