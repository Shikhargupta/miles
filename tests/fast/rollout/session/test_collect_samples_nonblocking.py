"""collect_samples must not assemble trajectories on the event loop.

The session server is one process with one event loop and every route shares it:
chat completions, delete, and collect. Assembling a long agentic trajectory is
seconds of straight-line CPU, so doing it inline stalls every other in-flight
request for the duration.

The stall is self-amplifying rather than merely slow. A client-side timeout on
collect marks the sample ABORTED, and the ``check_no_aborted`` dynamic-sampling
filter then rejects that sample's entire group, so the rollout relaunches a whole
group of episodes onto a server that is already behind.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.rollout.session.core import SessionCore

# Long enough that a blocking implementation is unambiguous, short enough to keep
# the test fast. Real assembly on a 76k-token trajectory is far slower than this.
_ASSEMBLE_SECONDS = 0.5


def _core_with_slow_assembly(monkeypatch):
    """A SessionCore whose CPU phase blocks for _ASSEMBLE_SECONDS.

    Patching _assemble_samples_payload (rather than the merge internals) keeps the
    test pinned to the contract under test: whatever that phase costs, it must not
    be paid on the event loop.
    """
    core = SessionCore.__new__(SessionCore)
    core.args = SimpleNamespace()

    session = SimpleNamespace(records=[object()], closing=False)
    registry = MagicMock()
    registry.get_session.return_value = session
    registry.tokenizer = MagicMock()
    core.registry = registry

    monkeypatch.setattr(SessionCore, "_session_metadata", lambda self, sid, s: {}, raising=True)

    def _slow_assemble(self, records, metadata, tokenizer, max_seq_len):
        time.sleep(_ASSEMBLE_SECONDS)  # stands in for token reconciliation + merge
        return b"payload"

    monkeypatch.setattr(SessionCore, "_assemble_samples_payload", _slow_assemble, raising=True)
    return core


@pytest.mark.asyncio
async def test_collect_samples_does_not_block_the_event_loop(monkeypatch):
    """A concurrent coroutine must keep getting scheduled while collect runs.

    Ticks are counted on a 10ms sleep. If assembly runs inline the loop is frozen
    and almost no ticks land; off the loop, ticks accrue throughout.
    """
    core = _core_with_slow_assembly(monkeypatch)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        response = await core.collect_samples("sid", max_seq_len=None)
    finally:
        beat.cancel()

    assert response.status_code == 200
    assert response.body == b"payload"

    # ~50 ticks are available across a 0.5s assembly. Require a clear majority so
    # the test fails loudly on the inline version (which yields ~0) without being
    # brittle about scheduler jitter.
    assert ticks >= 25, (
        f"event loop was starved during collect_samples: only {ticks} heartbeat ticks "
        f"in {_ASSEMBLE_SECONDS}s -- trajectory assembly is running inline on the loop"
    )


@pytest.mark.asyncio
async def test_collect_samples_all_truncated_still_returns_empty_payload(monkeypatch):
    """The None sentinel from the worker maps back to the all_truncated reply."""
    core = _core_with_slow_assembly(monkeypatch)
    monkeypatch.setattr(
        SessionCore,
        "_assemble_samples_payload",
        lambda self, records, metadata, tokenizer, max_seq_len: None,
        raising=True,
    )
    response = await core.collect_samples("sid", max_seq_len=8)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_collect_samples_validation_error_still_maps_to_422(monkeypatch):
    """Exceptions raised inside the worker thread must surface unchanged.

    asyncio.to_thread re-raises in the awaiting coroutine, so the existing
    AssertionError/ValueError -> 422 contract has to keep holding across the
    thread boundary.
    """
    core = _core_with_slow_assembly(monkeypatch)

    def _boom(self, records, metadata, tokenizer, max_seq_len):
        raise ValueError("bad records")

    monkeypatch.setattr(SessionCore, "_assemble_samples_payload", _boom, raising=True)
    response = await core.collect_samples("sid", max_seq_len=None)
    assert response.status_code == 422
    assert b"bad records" in response.body
