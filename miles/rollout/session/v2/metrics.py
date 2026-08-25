from dataclasses import asdict, dataclass, field, fields

from miles.utils.types import Sample

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
