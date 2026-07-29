import pytest

from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


class TestBaseWorkerHandle:
    def test_incomplete_implementation_rejected(self):
        """A handle that does not implement wait_ready cannot be instantiated."""

        class Incomplete(BaseWorkerHandle):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    async def test_complete_implementation_is_usable(self):
        """A handle implementing wait_ready can be instantiated and awaited."""

        class Ready(BaseWorkerHandle):
            async def wait_ready(self, *, timeout: float) -> None:
                pass

        await Ready().wait_ready(timeout=1.0)


class TestWorkerUnreachableError:
    def test_is_plain_exception_without_submission_state(self):
        """The error carries only its message and standard exception state."""
        error = WorkerUnreachableError("boom")

        assert str(error) == "boom"
        assert not hasattr(error, "submitted")
