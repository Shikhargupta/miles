"""
SWE-Agent: reward, metrics, and rollout class.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run

Task-type agnostic — reward is pre-computed by the Harbor environment
and stored in sample.metadata["reward"] regardless of task type.

Components:
  - reward_func: reads pre-computed reward from sample metadata
  - sanitize_aborted_samples: makes aborted trials safe to train
  - aggregate_agent_metrics: aggregates agent timing/count metrics
  - RolloutFn: InferenceRolloutFn subclass that sanitizes samples and logs agent metrics
"""

import copy
import logging

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the agent environment during generate().

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Aborted-sample sanitation --


def _is_trainable(sample: Sample) -> bool:
    return sample.response_length > 0 and sample.rollout_log_probs is not None


def sanitize_aborted_samples(groups: list[list[Sample]]) -> dict[str, int]:
    """Make aborted trials safe to train (no dynamic-sampling filter drops them).

    Aborted trials that still produced tokens train with reward 0. Trials that
    never recorded a model call (e.g. the sandbox failed to boot) have no
    tokens and ``rollout_log_probs=None``, which crashes the trainer, so each
    is replaced by a copy of a healthy sibling with ``remove_sample=True``:
    the batch keeps its shape while the copy's loss mask is zeroed out.
    """
    stats = {"reward_zero_filled": 0, "empty_replaced": 0, "empty_unreplaceable": 0}
    batch_donors = [s for group in groups for s in group if _is_trainable(s)]
    for group in groups:
        donors = [s for s in group if _is_trainable(s)] or batch_donors
        for i, sample in enumerate(group):
            if _is_trainable(sample):
                if sample.reward is None:
                    sample.reward = 0.0
                    stats["reward_zero_filled"] += 1
                continue
            if not donors:
                stats["empty_unreplaceable"] += 1
                continue
            replacement = copy.deepcopy(donors[0])
            replacement.group_index = sample.group_index
            replacement.index = sample.index
            replacement.prompt = sample.prompt
            replacement.label = sample.label
            replacement.metadata = sample.metadata
            replacement.reward = 0.0
            replacement.remove_sample = True
            group[i] = replacement
            stats["empty_replaced"] += 1
    return {key: count for key, count in stats.items() if count}


# -- Agent Metrics Aggregation --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    return [m.get(key, 0) for m in all_metrics]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    """Rollout function with agent metrics aggregation."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        sanitize_stats = sanitize_aborted_samples(output.samples)
        if sanitize_stats:
            logger.info(f"Sanitized aborted samples for rollout {input.rollout_id}: {sanitize_stats}")

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        agent_metrics = aggregate_agent_metrics(all_samples)
        if agent_metrics:
            metrics = output.metrics or {}
            metrics.update(agent_metrics)
            output.metrics = metrics
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")

        return output
