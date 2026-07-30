from __future__ import annotations

import pytest
from tests.fast.utils.workers.reconcile.utils import settle

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.retry_scheduler import RetryScheduler
from miles.utils.workers.reconcile.work_queue import WorkQueue


def make_scheduler(
    *, failure_base_delay: float = 1.0, failure_max_delay: float = 8.0
) -> tuple[RetryScheduler, WorkQueue, FakeClock]:
    queue = WorkQueue()
    clock = FakeClock()
    scheduler = RetryScheduler(
        queue=queue, failure_base_delay=failure_base_delay, failure_max_delay=failure_max_delay, clock=clock
    )
    return scheduler, queue, clock


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs", [dict(failure_base_delay=0.0), dict(failure_base_delay=-1.0), dict(failure_max_delay=0.5)]
    )
    def test_a_non_positive_or_inverted_delay_is_rejected(self, kwargs):
        """A zero base delay would retry in a hot loop; a max below the base is a contradiction."""
        with pytest.raises(AssertionError):
            make_scheduler(**kwargs)


class TestBackoff:
    async def test_consecutive_failures_double_the_delay_up_to_the_max(self):
        """The timer fires at base*2^(n-1), capped."""
        scheduler, queue, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=4.0)
        for expected_delay in (1.0, 2.0, 4.0, 4.0):
            scheduler.note_failure("cell-a")
            await clock.elapse(expected_delay - 0.5)
            assert "cell-a" not in queue._keys
            await clock.elapse(0.5)
            await settle()
            assert "cell-a" in queue._keys
            queue._keys.remove("cell-a")

    async def test_a_new_failure_replaces_the_pending_timer(self):
        """Latest-wins: the old timer is cancelled, only the new delay fires."""
        scheduler, queue, clock = make_scheduler(failure_base_delay=4.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")
        await clock.elapse(3.0)
        scheduler.note_failure("cell-a")

        await clock.elapse(1.5)
        await settle()
        assert "cell-a" not in queue._keys
        assert clock.pending_count == 1

        await clock.elapse(6.5)
        await settle()
        assert "cell-a" in queue._keys

    async def test_success_clears_the_count_and_cancels_the_timer(self):
        """A recovered key starts over at the base delay with no stale wakeup."""
        scheduler, queue, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")
        scheduler.note_failure("cell-a")
        scheduler.note_success("cell-a")
        await settle()

        assert scheduler._failures == {}
        await clock.elapse(100.0)
        await settle()
        assert "cell-a" not in queue._keys

        scheduler.note_failure("cell-a")
        await clock.elapse(1.0)
        await settle()
        assert "cell-a" in queue._keys

    async def test_failure_counts_are_per_key(self):
        """One key's failures never inflate another key's delay."""
        scheduler, queue, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")
        scheduler.note_failure("cell-a")
        scheduler.note_failure("cell-b")

        await clock.elapse(1.0)
        await settle()
        assert "cell-b" in queue._keys
        assert "cell-a" not in queue._keys


class TestShutdown:
    async def test_a_failure_after_shutdown_installs_no_timer(self):
        """Once shut down, the scheduler must not create new tasks."""
        scheduler, _, clock = make_scheduler()
        await scheduler.shutdown()
        scheduler.note_failure("cell-a")

        assert clock.pending_count == 0

    async def test_shutdown_cancels_a_pending_timer(self):
        """Shutdown owns its own tasks: a retry in flight must not outlive it or reach the queue."""
        scheduler, queue, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")

        await scheduler.shutdown()
        await clock.elapse(1.0)
        await settle()

        assert clock.pending_count == 0
        assert "cell-a" not in queue._keys
