"""Client-driven operation queue: per registration, one ordinal-keyed dict serves as reorder buffer, idempotency table, and result store."""

import hashlib
import json
from dataclasses import dataclass, field

DATA_KINDS = ("forward_backward", "forward")
CONTROL_KINDS = ("optim_step", "save_weights_for_sampler", "save_state", "load_state")


class QueueFull(RuntimeError):
    """Capacity reached; the wire layer maps this to 429 (sampling plane only, never training)."""


class BadRequest(ValueError):
    """Contract violation by the client; ValueError so the control-plane handler maps it to 400."""


def payload_fingerprint(kind: str, payload: dict) -> str:
    """Digest of identity-relevant content; callers must exclude volatile per-retry fields first."""
    canonical = json.dumps({"k": kind, "p": payload or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class OperationRecord:
    ordinal: int
    request_id: str
    kind: str
    payload: dict
    fingerprint: str
    status: str = "QUEUED"  # QUEUED|RUNNING|DONE|FAILED
    result: dict | None = None
    error: str | None = None
    error_kind: str | None = None  # user|server
    delivered: bool = False
    evicted: bool = False


@dataclass
class OperationQueue:
    """Per-registration ordinal queue; cap=None (training plane) because capacity-429 deadlocks the SDK, which posts chunk 1 last."""

    cap: int | None = None
    keep_delivered: int = 4
    ops: dict[int, OperationRecord] = field(default_factory=dict)
    by_request_id: dict[str, int] = field(default_factory=dict)
    next_to_run: int = 1
    poisoned: bool = False  # a forward_backward FAILED since the last optim_step terminal

    def __post_init__(self):
        # Retained delivered terminals consume capacity: keep_delivered >= cap wedges the queue permanently.
        if self.cap is not None and self.keep_delivered >= self.cap:
            raise ValueError("keep_delivered must be < cap (retained terminals consume capacity)")
