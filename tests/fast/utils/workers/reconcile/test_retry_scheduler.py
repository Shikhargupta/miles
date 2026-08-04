from __future__ import annotations

import pytest
from tests.fast.utils.workers.reconcile.utils import settle

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.retry_scheduler import RetryScheduler


def make_scheduler(
    *, failure_base_delay: float = 1.0, failure_max_delay: float = 8.0
) -> tuple[RetryScheduler[str], list[str], FakeClock]:
    retried: list[str] = []
    clock = FakeClock()
    scheduler: RetryScheduler[str] = RetryScheduler(
        on_retry=retried.append,
        failure_base_delay=failure_base_delay,
        failure_max_delay=failure_max_delay,
        clock=clock,
    )
    return scheduler, retried, clock


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
        scheduler, retried, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=4.0)
        for expected_delay in (1.0, 2.0, 4.0, 4.0):
            scheduler.note_failure("cell-a")
            await clock.elapse(expected_delay - 0.5)
            assert retried == []
            await clock.elapse(0.5)
            await settle()
            assert retried == ["cell-a"]
            retried.clear()

    async def test_a_new_failure_replaces_the_pending_deadline(self):
        """Latest-wins: the old deadline is overwritten, only the new delay fires."""
        scheduler, retried, clock = make_scheduler(failure_base_delay=4.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")
        await clock.elapse(3.0)
        scheduler.note_failure("cell-a")

        await clock.elapse(1.5)
        await settle()
        assert retried == []
        assert clock.pending_count == 1

        await clock.elapse(6.5)
        await settle()
        assert retried == ["cell-a"]

    async def test_success_clears_the_count_and_the_pending_retry(self):
        """A recovered key starts over at the base delay with no stale wakeup."""
        scheduler, retried, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")
        scheduler.note_failure("cell-a")
        scheduler.note_success("cell-a")
        await settle()

        assert scheduler._infos == {}
        await clock.elapse(100.0)
        await settle()
        assert retried == []

        scheduler.note_failure("cell-a")
        await clock.elapse(1.0)
        await settle()
        assert retried == ["cell-a"]

    async def test_an_idle_scheduler_holds_no_wakeup(self):
        """With nothing due, the poller parks on its event rather than on the clock."""
        scheduler, retried, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")

        await clock.elapse(1.0)
        await settle()

        assert retried == ["cell-a"]
        assert clock.pending_count == 0

    async def test_failure_counts_are_per_key(self):
        """One key's failures never inflate another key's delay."""
        scheduler, retried, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")
        scheduler.note_failure("cell-a")
        scheduler.note_failure("cell-b")

        await clock.elapse(1.0)
        await settle()
        assert retried == ["cell-b"]


class TestShutdown:
    async def test_a_failure_after_shutdown_schedules_no_retry(self):
        """Once shut down, the scheduler must not schedule anything again."""
        scheduler, _, clock = make_scheduler()
        await scheduler.shutdown()
        scheduler.note_failure("cell-a")

        assert clock.pending_count == 0

    async def test_shutdown_cancels_the_poller(self):
        """Shutdown owns its own task: a retry in flight must not outlive it or reach the consumer."""
        scheduler, retried, clock = make_scheduler(failure_base_delay=1.0, failure_max_delay=64.0)
        scheduler.note_failure("cell-a")

        await scheduler.shutdown()
        await clock.elapse(1.0)
        await settle()

        assert clock.pending_count == 0
        assert retried == []
