import logging
from argparse import Namespace
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.errors import TokenizationError
from miles.rollout.session.samples.merge import (
    compute_samples_from_openai_records,
    merge_samples_with_addition_r3,
    truncate_samples_by_total_tokens,
)
from miles.rollout.session.v2.session_state import SessionRegistryV2, SessionStateV2
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NODE_ADDITIVE_METRICS_METADATA_KEY = "_node_additive_metrics"


def _sum_flat_metric(
    left: Sample.SpecInfo | Sample.PrefixCacheInfo,
    right: Sample.SpecInfo | Sample.PrefixCacheInfo,
) -> Sample.SpecInfo | Sample.PrefixCacheInfo:
    assert type(left) is type(right)
    left_counters = asdict(left)
    right_counters = asdict(right)
    assert left_counters.keys() == right_counters.keys()
    return type(left)(**{name: value + right_counters[name] for name, value in left_counters.items()})


@dataclass(frozen=True)
class AdditiveNodeMetrics:
    """Explicit registry of metrics attributed exactly once per tree node."""

    spec_info: Sample.SpecInfo = field(default_factory=Sample.SpecInfo)
    prefix_cache_info: Sample.PrefixCacheInfo = field(default_factory=Sample.PrefixCacheInfo)

    @classmethod
    def from_sample(cls, sample: Sample) -> "AdditiveNodeMetrics":
        return cls(spec_info=sample.spec_info, prefix_cache_info=sample.prefix_cache_info)

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> "AdditiveNodeMetrics":
        metric_names = {metric.name for metric in fields(cls)}
        assert set(data) == metric_names, f"node metric mismatch: expected {metric_names}, got {set(data)}"

        for metric_name, metric_type in (
            ("spec_info", Sample.SpecInfo),
            ("prefix_cache_info", Sample.PrefixCacheInfo),
        ):
            counter_names = {counter.name for counter in fields(metric_type)}
            counters = data[metric_name]
            assert (
                set(counters) == counter_names
            ), f"{metric_name} counter mismatch: expected {counter_names}, got {set(counters)}"
        return cls(
            spec_info=Sample.SpecInfo(**data["spec_info"]),
            prefix_cache_info=Sample.PrefixCacheInfo(**data["prefix_cache_info"]),
        )

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "spec_info": self.spec_info.to_dict(),
            "prefix_cache_info": self.prefix_cache_info.to_dict(),
        }

    def __add__(self, other: "AdditiveNodeMetrics") -> "AdditiveNodeMetrics":
        return type(self)(
            spec_info=_sum_flat_metric(self.spec_info, other.spec_info),
            prefix_cache_info=_sum_flat_metric(self.prefix_cache_info, other.prefix_cache_info),
        )

    def assign_to(self, sample: Sample) -> None:
        sample.spec_info = self.spec_info
        sample.prefix_cache_info = self.prefix_cache_info


def tree_metadata(state: SessionStateV2) -> dict:
    """The structural layer: node and leaf tables, index-aligned with commits.

    ``response_id`` is the branch<->leaf join key — the agent saw the same id
    in each chat response, so the semantic layer can key per-branch data on it.
    """
    nodes = [
        {
            "id": node.seq,
            "parent": node.parent.seq if node.parent is not None else None,
            "seq": node.seq,
            "truncated": node.truncated,
            "committed_at": node.committed_at,
            "completion_span": list(node.completion_span),
            "num_tokens": len(node.token_ids),
            "response_id": node.response_id,
        }
        for node in state.tree.nodes
    ]
    leaves = [
        {"node_id": leaf.seq, "path_node_ids": [n.seq for n in leaf.path_nodes()]} for leaf in state.tree.leaves()
    ]
    return {"nodes": nodes, "leaves": leaves}


def build_leaf_material(
    args: Namespace,
    state: SessionStateV2,
    registry: SessionRegistryV2,
    *,
    session_id: str,
    max_seq_len: int | None,
    use_addition_r3: bool = False,
) -> list[Sample]:
    """Merge each leaf's root-to-leaf node records into one raw sample, in commit order.

    Leaves whose turns all truncate away are dropped. Each sample's metadata
    carries the ``leaf`` descriptor plus the flat TITO bookkeeping keys for
    the downstream pick/post-process hooks.

    With ``use_addition_r3``, each record along a path carries only its
    additional R3 rows; the per-leaf assembler materializes the required prefix
    because a path is the linear record chain its offsets were computed on.
    """
    material: list[Sample] = []
    for leaf in state.tree.leaves():
        path = leaf.path_nodes()
        records = [node.record for node in path]
        turns = compute_samples_from_openai_records(
            args,
            records,
            registry.tokenizer,
            accumulated_token_ids=leaf.token_ids,
            max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            use_addition_r3=use_addition_r3,
        )
        if max_seq_len is not None:
            turns = truncate_samples_by_total_tokens(turns, max_seq_len, registry.tokenizer)
        if not turns:
            continue
        if use_addition_r3:
            sample = merge_samples_with_addition_r3(args, turns, records, registry.tokenizer)
        else:
            sample = merge_samples(turns, registry.tokenizer)
        # `merge_samples` may stop before the last turn on a status or replay gap;
        # keep counters only for the source-turn prefix represented by `sample`.
        merged_turn_indexes = [i for i, turn in enumerate(turns) if turn.tokens == sample.tokens]
        assert merged_turn_indexes, "merged sample must end at one of its source turns"
        merged_turn_count = merged_turn_indexes[0] + 1
        tools = path[-1].record.request.get("tools")
        flat: dict[str, Any] = {
            "accumulated_token_ids": list(leaf.token_ids),
            NODE_ADDITIVE_METRICS_METADATA_KEY: [
                {"node_id": node.seq, "metrics": AdditiveNodeMetrics.from_sample(turn).to_dict()}
                for node, turn in zip(path[:merged_turn_count], turns[:merged_turn_count], strict=True)
            ],
            "leaf": {
                "node_id": leaf.seq,
                "parent": leaf.parent.seq if leaf.parent is not None else None,
                "path_node_ids": [n.seq for n in path],
                "response_id": leaf.response_id,
            },
        }
        try:
            mismatch = registry.compute_mismatch(leaf.path_messages(), leaf.token_ids, tools)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            flat["tito_session_mismatch"] = mismatch
        sample.metadata = {**(sample.metadata or {}), **flat}
        material.append(sample)
    return material
