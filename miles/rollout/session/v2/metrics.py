from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.session.v2.tree_trajectory import TrajectoryNode

SESSION_ROLLOUT_METRICS_KEY = "session_rollout_metrics"

_ENVELOPE_FIELDS = frozenset({"session_id", "available", "metrics"})
_SPEC_INFO_FIELDS = frozenset(
    {
        "spec_num_correct_drafts",
        "spec_num_proposed_drafts",
        "spec_verify_ct",
        "completion_tokens",
    }
)

def build_session_rollout_metrics(args: Namespace, session_id: str, nodes: list[TrajectoryNode]) -> dict:
    spec_info = Sample.SpecInfo()
    if args.sglang_speculative_algorithm:
        for node in nodes:
            spec_info.add(node.record.response["choices"][0]["meta_info"])
    return {
        "session_id": session_id,
        "available": True,
        "metrics": {"spec_info": spec_info.to_dict()},
    }


def _require_exact_object(value: object, name: str, expected_fields: frozenset[str] | set[str]) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    if set(value) != set(expected_fields):
        actual_fields = sorted(str(field) for field in value)
        raise ValueError(f"{name} fields must be exactly {sorted(expected_fields)}, got {actual_fields}")
    return value


def _validate_metrics(metrics: object) -> dict:
    metrics = _require_exact_object(metrics, "session_rollout_metrics.metrics", {"spec_info"})
    payload_name = "session_rollout_metrics.metrics.spec_info"
    payload = _require_exact_object(metrics["spec_info"], payload_name, _SPEC_INFO_FIELDS)
    for field, value in payload.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{payload_name}.{field} must be a non-negative integer")
    return metrics


def _validate_envelope(envelope: object) -> dict:
    envelope = _require_exact_object(envelope, "session_rollout_metrics", _ENVELOPE_FIELDS)
    session_id = envelope["session_id"]
    if type(session_id) is not str or not session_id:
        raise ValueError("session_rollout_metrics.session_id must be a non-empty string")
    if type(envelope["available"]) is not bool:
        raise ValueError("session_rollout_metrics.available must be a boolean")
    return envelope


def read_server_session_rollout_metrics(session_metadata: dict, expected_session_id: str) -> dict:
    if type(session_metadata) is not dict:
        raise ValueError("session metadata must be an object")
    if SESSION_ROLLOUT_METRICS_KEY not in session_metadata:
        raise ValueError(f"session metadata is missing {SESSION_ROLLOUT_METRICS_KEY}")
    envelope = _validate_envelope(session_metadata[SESSION_ROLLOUT_METRICS_KEY])
    if envelope["session_id"] != expected_session_id:
        raise ValueError(
            "session_rollout_metrics.session_id does not match the collected session: "
            f"{envelope['session_id']!r} != {expected_session_id!r}"
        )
    if not envelope["available"] or envelope["metrics"] is None:
        raise ValueError("a successful session collect must carry available metrics")
    return _validate_metrics(envelope["metrics"])


def assign_session_rollout_metrics(
    samples: list[Sample], *, session_id: str, available: bool, metrics: dict | None
) -> None:
    if type(session_id) is not str or not session_id:
        raise ValueError("session_rollout_metrics.session_id must be a non-empty string")
    if type(available) is not bool:
        raise ValueError("session_rollout_metrics.available must be a boolean")
    if available:
        metrics = _validate_metrics(metrics)
    elif metrics is not None:
        raise ValueError("unavailable session_rollout_metrics must not carry metrics")

    for sample in samples:
        sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
        sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = {
            "session_id": session_id,
            "available": available,
            "metrics": metrics,
        }


def read_sample_rollout_metrics(samples: list[Sample]) -> tuple[dict | None, str]:
    """Read the shared session metrics from one rollout's training samples."""
    envelopes = []
    for sample in samples:
        if SESSION_ROLLOUT_METRICS_KEY not in sample.metadata:
            raise ValueError(f"v2 sample metadata is missing {SESSION_ROLLOUT_METRICS_KEY}")
        envelopes.append(_validate_envelope(sample.metadata[SESSION_ROLLOUT_METRICS_KEY]))

    session_ids = {envelope["session_id"] for envelope in envelopes}
    if len(session_ids) != 1:
        raise ValueError(f"one rollout has inconsistent session IDs: {sorted(session_ids)}")
    session_id = next(iter(session_ids))

    availability = {envelope["available"] for envelope in envelopes}
    if len(availability) != 1:
        raise ValueError(f"session {session_id!r} has inconsistent metrics availability")
    if not envelopes[0]["available"]:
        if any(envelope["metrics"] is not None for envelope in envelopes):
            raise ValueError(f"unavailable session {session_id!r} must not carry metrics")
        return None, session_id

    metrics = [_validate_metrics(envelope["metrics"]) for envelope in envelopes]
    if any(value != metrics[0] for value in metrics[1:]):
        raise ValueError(f"session {session_id!r} has inconsistent metrics across compacted samples")
    return metrics[0], session_id
