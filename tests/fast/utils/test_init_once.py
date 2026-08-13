import inspect

import pytest

from miles.utils.context_lock import ContextLock, enforce_lock_discipline, lock_exempt
from miles.utils.init_once import InitOnce, init_once_guarded


class TestInitOnce:
    def test_a_fresh_component_reports_itself_uninitialized(self):
        """A restarted orchestration script decides how to start out of exactly this answer."""
        assert InitOnce(component="Widget").is_initialized is False

    def test_a_component_still_inside_its_init_is_not_initialized_yet(self):
        """Marking a component initialized before it built anything is what hid a half-built fleet."""
        once = InitOnce(component="Widget")

        with once.guard():
            assert once.is_initialized is False
            with pytest.raises(AssertionError, match="Widget is initializing"):
                with once.guard():
                    pass

    def test_a_completed_init_marks_the_component_initialized(self):
        """The take-over path has to see the component the previous script built as built."""
        once = InitOnce(component="Widget")

        with once.guard():
            pass

        assert once.is_initialized is True

    def test_an_init_that_raised_is_reported_as_failed_and_never_as_initialized(self):
        """A controller that died before creating its servers must not be taken over as a live one."""
        once = InitOnce(component="InferenceController")

        with pytest.raises(RuntimeError, match="boom"):
            with once.guard():
                raise RuntimeError("boom")

        assert once.is_initialized is False

    def test_a_second_init_in_one_process_fails_loudly_and_names_the_component(self):
        """Re-initializing a live system behind the back of whoever drives it is the bug this exists for."""
        once = InitOnce(component="TrainerController(actor)")
        with once.guard():
            pass

        with pytest.raises(AssertionError, match=r"TrainerController\(actor\) is complete"):
            with once.guard():
                pass

    def test_initializing_a_component_whose_init_failed_is_refused(self):
        """A failed init leaves state nobody can reason about, so the pod has to be replaced instead."""
        once = InitOnce(component="Widget")
        with pytest.raises(RuntimeError):
            with once.guard():
                raise RuntimeError("boom")

        with pytest.raises(AssertionError, match="Widget is failed"):
            with once.guard():
                pass

    def test_asserting_initialized_refuses_a_component_that_never_ran_init(self):
        """load_state reloads state that init built, so it cannot run on a component without it."""
        with pytest.raises(AssertionError, match="not started, not initialized"):
            InitOnce(component="Widget").assert_initialized()

    def test_asserting_initialized_passes_after_init(self):
        """The resume path runs on exactly this state and must not be blocked by its own guard."""
        once = InitOnce(component="Widget")
        with once.guard():
            pass

        once.assert_initialized()


class _AsyncWorker:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self._init_once = InitOnce(component="AsyncWorker")
        self._failure = failure
        self.calls = 0

    @init_once_guarded
    async def init(self, answer: int = 7) -> int:
        self.calls += 1
        if self._failure is not None:
            raise self._failure
        return answer


class _SyncWorker:
    def __init__(self) -> None:
        self._init_once = InitOnce(component="SyncWorker")
        self.calls = 0

    @init_once_guarded
    def init(self) -> str:
        self.calls += 1
        return "built"


@enforce_lock_discipline
class _LockDisciplinedWorker:
    @lock_exempt
    def __init__(self) -> None:
        self.context_lock = ContextLock("LockDisciplinedWorker")
        self._init_once = InitOnce(component="LockDisciplinedWorker")

    @lock_exempt
    @init_once_guarded
    async def init(self) -> None:
        return None


class TestInitOnceGuarded:
    async def test_an_async_init_runs_once_and_answers_what_it_returned(self):
        """The decorator replaces a hand-written guard, so the wrapped init must stay a plain init."""
        worker = _AsyncWorker()

        assert await worker.init(3) == 3
        assert (worker.calls, worker._init_once.is_initialized) == (1, True)

    def test_a_sync_init_is_guarded_too(self):
        """RolloutExecutor's sibling components init synchronously, and they need the same fence."""
        worker = _SyncWorker()

        assert worker.init() == "built"
        assert worker._init_once.is_initialized is True

    async def test_a_second_init_is_refused_without_re_entering_the_body(self):
        """A restarted script must resume a built component, never build it a second time behind the first driver."""
        worker = _AsyncWorker()
        await worker.init()

        with pytest.raises(AssertionError, match="AsyncWorker is complete"):
            await worker.init()

        assert worker.calls == 1

    async def test_an_init_that_raised_leaves_the_component_failed(self):
        """A half-built component has to be replaced, so the guard must not report it as initialized or retryable."""
        worker = _AsyncWorker(failure=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await worker.init()

        assert worker._init_once.is_initialized is False
        with pytest.raises(AssertionError, match="AsyncWorker is failed"):
            await worker.init()

    def test_the_wrapper_keeps_the_signature_the_rpc_surface_reads(self):
        """collect_rpc_method_specs unwraps and types the real parameters; a lost signature breaks the wire."""
        assert str(inspect.signature(inspect.unwrap(_AsyncWorker.init))) == "(self, answer: int = 7) -> int"
        assert _AsyncWorker.init.__name__ == "init"

    def test_stacking_under_lock_exempt_keeps_the_context_lock_marker(self):
        """enforce_lock_discipline reads the marker off the class member, and an unmarked init fails at import."""
        assert getattr(_LockDisciplinedWorker.init, "_context_lock_discipline", None) == "lock_exempt"
