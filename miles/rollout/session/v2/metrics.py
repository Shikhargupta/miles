from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.session.v2.tree_trajectory import TrajectoryNode


def assign_node_metrics_to_sample_0(args: Namespace, samples: list[Sample], nodes: list[TrajectoryNode]) -> None:
    """Assign every committed generation's telemetry exactly once.

    The first returned sample is only a wire carrier. Consumers must aggregate
    these fields across the returned samples; carrier identity has no meaning.
    """
    spec_info = Sample.SpecInfo()
    prefix_cache_info = Sample.PrefixCacheInfo()
    for node in nodes:
        meta_info = node.record.response["choices"][0]["meta_info"]
        if args.sglang_speculative_algorithm:
            spec_info.add(meta_info)
        prefix_cache_info.add(meta_info)

    for sample in samples:
        sample.spec_info = Sample.SpecInfo()
        sample.prefix_cache_info = Sample.PrefixCacheInfo()
    if samples:
        samples[0].spec_info = spec_info
        samples[0].prefix_cache_info = prefix_cache_info
