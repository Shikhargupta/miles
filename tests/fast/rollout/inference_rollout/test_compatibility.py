import asyncio
from unittest.mock import MagicMock

import pytest

from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.inference_rollout.compatibility import (
    LegacyGenerateFnAdapter,
    LegacyRolloutFnAdapter,
    call_rollout_function,
    call_rollout_function_async,
    load_generate_function,
    load_rollout_function,
)
from miles.utils.async_utils import run
from miles.utils.misc import function_registry


@pytest.fixture
def constructor_input():
    return RolloutFnConstructorInput(args="dummy_args", data_source="dummy_data_source")


@pytest.fixture
def make_generate_fn_input():
    def _make(evaluation: bool = False):
        state = MagicMock()
        state.args = MagicMock()

        return GenerateFnInput(
            state=state,
            sample={"text": "test prompt"},
            sampling_params={"temperature": 0.7},
            evaluation=evaluation,
        )

    return _make


class TestSupportedRolloutFormats:
    """
    Documentation test to show various supported rollout function formats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_raw_output(self, constructor_input, evaluation):
        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            if evaluation:
                return {"metric": {"accuracy": 0.9}}
            return [[{"text": "sample"}]]

        with function_registry.temporary("test:legacy_rollout", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_rollout")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, LegacyRolloutFnAdapter)
            if evaluation:
                assert isinstance(result, RolloutFnEvalOutput)
                assert result.data == {"metric": {"accuracy": 0.9}}
            else:
                assert isinstance(result, RolloutFnTrainOutput)
                assert result.samples == [[{"text": "sample"}]]

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_typed_output(self, constructor_input, evaluation):
        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            if evaluation:
                return RolloutFnEvalOutput(data={"ds": {"acc": 0.95}})
            return RolloutFnTrainOutput(samples=[[{"text": "typed"}]])

        with function_registry.temporary("test:legacy_typed", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_typed")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            if evaluation:
                assert isinstance(result, RolloutFnEvalOutput)
                assert result.data == {"ds": {"acc": 0.95}}
            else:
                assert isinstance(result, RolloutFnTrainOutput)
                assert result.samples == [[{"text": "typed"}]]

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_sync_class(self, constructor_input, evaluation):
        class SyncRolloutFn:
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            def __call__(self, input):
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"test": {"score": 1}})
                return RolloutFnTrainOutput(samples=[[{"text": "sync"}]])

        with function_registry.temporary("test:sync_class", SyncRolloutFn):
            fn = load_rollout_function(constructor_input, "test:sync_class")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, SyncRolloutFn)
            expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_async_class(self, constructor_input, evaluation):
        class AsyncRolloutFn:
            def __init__(self, input: RolloutFnConstructorInput):
                pass

            async def __call__(self, input):
                await asyncio.sleep(0.001)
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"benchmark": {"accuracy": 0.98}})
                return RolloutFnTrainOutput(samples=[[{"text": "async"}]])

        with function_registry.temporary("test:async_class", AsyncRolloutFn):
            fn = load_rollout_function(constructor_input, "test:async_class")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, AsyncRolloutFn)
            expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
            assert isinstance(result, expected_type)


class TestAsyncInvocation:
    """``call_rollout_function_async`` — the manager's invocation path
    (external review 0813 §4.2/§6.4): async rollout fns are awaited DIRECTLY
    on the caller's loop so cancellation reaches the coroutine; only sync fns
    ride a worker thread. Never a thread + the background loop for a
    coroutine — that detached it from its caller."""

    def test_async_class_runs_on_the_callers_loop(self):
        loops = []

        class AsyncRolloutFn:
            async def __call__(self, input):
                loops.append(asyncio.get_running_loop())
                return RolloutFnTrainOutput(samples=[[{"text": "async"}]])

        async def scenario():
            result = await call_rollout_function_async(AsyncRolloutFn(), RolloutFnTrainInput(rollout_id=1))
            assert result.samples == [[{"text": "async"}]]
            assert loops == [asyncio.get_running_loop()]  # the SAME loop, no bridge

        asyncio.run(scenario())

    def test_cancelling_the_caller_cancels_the_rollout_coroutine(self):
        """The 0813 review's reproduction, inverted: cancelling the invoking
        task must cancel the rollout coroutine itself — it must never keep
        running (and mutating external state) after its caller is gone."""
        started = asyncio.Event()
        observed = {"cancelled": False, "completed": False}

        class BlockingAsyncRollout:
            async def __call__(self, input):
                started.set()
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    observed["cancelled"] = True
                    raise
                observed["completed"] = True
                return RolloutFnTrainOutput(samples=[])

        async def scenario():
            task = asyncio.create_task(
                call_rollout_function_async(BlockingAsyncRollout(), RolloutFnTrainInput(rollout_id=1))
            )
            await asyncio.wait_for(started.wait(), timeout=2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert observed["cancelled"] is True
            assert observed["completed"] is False

        asyncio.run(scenario())

    def test_sync_class_runs_in_a_worker_thread(self):
        import threading

        threads = []

        class SyncRolloutFn:
            def __call__(self, input):
                threads.append(threading.current_thread())
                return RolloutFnTrainOutput(samples=[[{"text": "sync"}]])

        async def scenario():
            result = await call_rollout_function_async(SyncRolloutFn(), RolloutFnTrainInput(rollout_id=1))
            assert result.samples == [[{"text": "sync"}]]
            # Off the event-loop thread: a blocking legacy fn cannot stall it.
            assert threads[0] is not threading.main_thread()

        asyncio.run(scenario())

    def test_sync_callable_returning_a_coroutine_awaits_on_the_callers_loop(self):
        loops = []

        def hybrid_fn(input):
            async def inner():
                loops.append(asyncio.get_running_loop())
                return RolloutFnTrainOutput(samples=[[{"text": "hybrid"}]])

            return inner()

        async def scenario():
            result = await call_rollout_function_async(hybrid_fn, RolloutFnTrainInput(rollout_id=1))
            assert result.samples == [[{"text": "hybrid"}]]
            assert loops == [asyncio.get_running_loop()]

        asyncio.run(scenario())


class TestSupportedGenerateFormats:
    """
    Documentation test similar to TestSupportedRolloutFormats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_with_evaluation_param(self, make_generate_fn_input, evaluation):
        async def legacy_generate_fn(args, sample, sampling_params, evaluation=False):
            return "my_sample"

        with function_registry.temporary("test:legacy_gen_eval", legacy_generate_fn):
            fn = load_generate_function("test:legacy_gen_eval")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, LegacyGenerateFnAdapter)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_without_evaluation_param(self, make_generate_fn_input, evaluation):
        async def legacy_generate_fn(args, sample, sampling_params):
            return "my_sample"

        with function_registry.temporary("test:legacy_gen", legacy_generate_fn):
            fn = load_generate_function("test:legacy_gen")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, LegacyGenerateFnAdapter)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_new_async_function_api(self, make_generate_fn_input, evaluation):
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            return GenerateFnOutput(samples="my_sample")

        with function_registry.temporary("test:new_async", generate):
            fn = load_generate_function("test:new_async")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_new_class_api(self, make_generate_fn_input, evaluation):
        class MyGenerateFn:
            async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
                return GenerateFnOutput(samples="my_sample")

        with function_registry.temporary("test:new_class", MyGenerateFn):
            fn = load_generate_function("test:new_class")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, MyGenerateFn)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"
