from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.loss_hub import logit_processors
from miles.backends.training_utils.loss_hub.math_utils import _calculate_log_probs_and_entropy_true_on_policy
from miles.backends.training_utils.sampling_mask import (
    RolloutSamplingMask,
    build_local_sampling_mask,
    get_rollout_sampling_mask,
)


def test_rollout_sampling_mask_from_supports_roundtrip():
    supports = [[5, 9], [2], [7, 8, 11]]

    mask = RolloutSamplingMask.from_supports(supports)

    assert len(mask) == 3
    assert mask.ids.tolist() == [5, 9, 2, 7, 8, 11]
    assert mask.offsets.tolist() == [0, 2, 3, 6]
    assert [support.tolist() for support in mask] == supports
    assert mask[-1].tolist() == [7, 8, 11]


def test_rollout_sampling_mask_accepts_plain_integer_lists():
    mask = RolloutSamplingMask(ids=[1, 3, 4], offsets=[0, 2, 3])

    assert isinstance(mask.ids, torch.Tensor)
    assert mask[1].tolist() == [4]


def test_rollout_sampling_mask_rejects_offsets_not_spanning_ids():
    with pytest.raises(ValueError, match="offsets must start at zero and end at the flattened id count"):
        RolloutSamplingMask(ids=[1, 2, 3], offsets=[0, 2])


def test_rollout_sampling_mask_rejects_empty_support():
    with pytest.raises(ValueError, match="non-empty sampling support"):
        RolloutSamplingMask(ids=[1, 2], offsets=[0, 2, 2])


def test_rollout_sampling_mask_rejects_non_integer_ids():
    with pytest.raises(ValueError, match="one-dimensional integers"):
        RolloutSamplingMask(ids=torch.tensor([0.5, 1.5]), offsets=[0, 2])


def test_rollout_sampling_mask_rejects_out_of_range_token_index():
    mask = RolloutSamplingMask.from_supports([[1], [2]])

    with pytest.raises(IndexError, match="out of range"):
        mask[2]


def test_get_rollout_sampling_mask_wraps_per_sample_values():
    batch = {
        "rollout_sampling_mask_ids": [[5, 9, 2], torch.tensor([7, 8])],
        "rollout_sampling_mask_offsets": [[0, 2, 3], torch.tensor([0, 2])],
    }

    masks = get_rollout_sampling_mask(batch)

    assert [len(mask) for mask in masks] == [2, 1]
    assert masks[0][0].tolist() == [5, 9]
    assert masks[1][0].tolist() == [7, 8]


def test_get_rollout_sampling_mask_fails_when_required_support_is_missing():
    with pytest.raises(ValueError, match="truncated-sampling actor scoring requires"):
        get_rollout_sampling_mask({})


def test_get_rollout_sampling_mask_fails_on_mismatched_sample_counts():
    with pytest.raises(ValueError, match="must cover the same samples"):
        get_rollout_sampling_mask(
            {
                "rollout_sampling_mask_ids": [[1]],
                "rollout_sampling_mask_offsets": [[0, 1], [0, 1]],
            }
        )


def test_build_local_sampling_mask_selects_original_response_rows_and_tp_shard():
    logits = torch.zeros(2, 4)
    mask = build_local_sampling_mask(
        logits,
        RolloutSamplingMask(ids=[1, 3, 4, 2, 5, 6, 7], offsets=[0, 2, 4, 7]),
        response_indices=[2, 0],
        response_length=3,
        tp_rank=1,
    )

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, False, False],
            ]
        ),
    )


def test_build_local_sampling_mask_requires_one_support_per_response_token():
    with pytest.raises(ValueError, match="covers 2 response tokens != response length 3"):
        build_local_sampling_mask(
            torch.zeros(1, 4),
            RolloutSamplingMask(ids=[0, 1], offsets=[0, 1, 2]),
            response_indices=[0],
            response_length=3,
            tp_rank=0,
        )


def test_build_local_sampling_mask_rejects_out_of_range_response_index():
    with pytest.raises(ValueError, match=r"response indices must be in \[0, 1\)"):
        build_local_sampling_mask(
            torch.zeros(1, 4),
            RolloutSamplingMask(ids=[0], offsets=[0, 1]),
            response_indices=[1],
            response_length=1,
            tp_rank=0,
        )


def test_true_on_policy_masks_logprob_but_keeps_full_vocab_entropy():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], requires_grad=True)
    tokens = torch.tensor([0])
    sampling_mask = torch.tensor([[True, False, True, False]])

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        sampling_mask=sampling_mask,
    )

    expected_masked_logprob = torch.log_softmax(logits.masked_fill(~sampling_mask, float("-inf")), dim=-1)[0, 0]
    full_log_probs = torch.log_softmax(logits, dim=-1)
    expected_entropy = -(full_log_probs.exp() * full_log_probs).sum(dim=-1)
    torch.testing.assert_close(log_probs, expected_masked_logprob.unsqueeze(0))
    torch.testing.assert_close(entropy, expected_entropy)


def _patched_single_rank_state(monkeypatch):
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, group=None),
        cp=SimpleNamespace(rank=0, size=1),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    return SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=False,
        fp16=False,
        log_probs_chunk_size=-1,
        vocab_size=4,
        allgather_cp=False,
    )


def test_get_log_probs_and_entropy_applies_per_response_sampling_support(monkeypatch):
    args = _patched_single_rank_state(monkeypatch)
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    result = logit_processors.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([2, 0, 3])],
        total_lengths=[3],
        response_lengths=[2],
        rollout_sampling_mask=[RolloutSamplingMask.from_supports([[0, 2], [1, 3]])],
    )

    expected = torch.stack(
        [
            torch.log_softmax(logits[0, 0, [0, 2]], dim=-1)[0],
            torch.log_softmax(logits[0, 1, [1, 3]], dim=-1)[1],
        ]
    )
    torch.testing.assert_close(result["log_probs"][0], expected)


def test_get_log_probs_and_entropy_rejects_mismatched_mask_batch(monkeypatch):
    args = _patched_single_rank_state(monkeypatch)

    with pytest.raises(ValueError, match="masks=2, responses=1"):
        logit_processors.get_log_probs_and_entropy(
            torch.zeros(1, 3, 4),
            args=args,
            unconcat_tokens=[torch.tensor([2, 0, 3])],
            total_lengths=[3],
            response_lengths=[2],
            rollout_sampling_mask=[RolloutSamplingMask.from_supports([[0]])] * 2,
        )


@pytest.mark.parametrize(("cp_rank", "expected_indices"), [(0, [0, 1]), (1, [2, 3])])
def test_allgather_cp_response_rows_keep_global_response_indices(monkeypatch, cp_rank, expected_indices):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=True,
    )

    response_chunks = list(
        logit_processors._iter_response_chunks(
            torch.zeros(1, 3, 4),
            args=args,
            unconcat_tokens=[torch.arange(6)],
            total_lengths=[6],
            response_lengths=[4],
            include_response_indices=True,
        )
    )
    logits_chunk, tokens_chunk, response_indices = response_chunks[0]

    assert list(response_indices) == expected_indices
    assert tokens_chunk.tolist() == [2 + index for index in expected_indices]
    assert logits_chunk.size(0) == len(expected_indices)


@pytest.mark.parametrize(("cp_rank", "expected_indices"), [(0, [4]), (1, [0, 1, 2, 3])])
def test_zigzag_cp_response_rows_keep_global_response_indices(monkeypatch, cp_rank, expected_indices):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=False,
    )

    response_chunks = list(
        logit_processors._iter_response_chunks(
            torch.zeros(1, 4, 4),
            args=args,
            unconcat_tokens=[torch.arange(8)],
            total_lengths=[8],
            response_lengths=[5],
            include_response_indices=True,
        )
    )
    logits_chunk, tokens_chunk, response_indices = response_chunks[0]

    assert list(response_indices) == expected_indices
    assert tokens_chunk.tolist() == [3 + index for index in expected_indices]
    assert logits_chunk.size(0) == len(expected_indices)
