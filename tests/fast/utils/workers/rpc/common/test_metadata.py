import functools
from typing import Any

import pytest
from pydantic import ValidationError

from tests.fast.utils.workers.rpc.common.postponed_annotation_worker import LatePayload, PostponedWorker

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import DEFAULT_CONCURRENCY_GROUP, collect_rpc_method_specs, rpc


class _Payload(StrictBaseModel):
    text: str
    count: int = 1


class _GoodWorker:
    demo_class_attribute = 3

    def demo_default_arg(self, a: int, b: int = 10) -> int:
        return a + b

    async def demo_async_model(self, payload: _Payload) -> _Payload:
        return payload

    @rpc(concurrency_group="heavy")
    def demo_grouped(self, step: int) -> None:
        pass

    @classmethod
    def demo_classmethod(cls, x: int) -> int:
        return x

    @staticmethod
    def demo_staticmethod(x: int) -> int:
        return x

    @property
    def demo_property(self) -> int:
        return 1

    def _demo_private(self, x):
        pass


class TestCollectSpecs:
    def test_collects_public_methods_only(self):
        """Public methods are collected; underscore-prefixed ones are skipped."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert set(specs) == {"demo_default_arg", "demo_async_model", "demo_grouped"}

    def test_non_instance_method_members_are_skipped(self):
        """Classmethods, staticmethods and properties are skipped like plain attributes."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert {"demo_classmethod", "demo_staticmethod", "demo_property", "demo_class_attribute"}.isdisjoint(specs)

    def test_default_concurrency_group(self):
        """Undecorated methods fall into the default concurrency group."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].concurrency_group == DEFAULT_CONCURRENCY_GROUP

    def test_decorated_concurrency_group(self):
        """@rpc(concurrency_group=...) is picked up by introspection."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_grouped"].concurrency_group == "heavy"

    def test_is_async_flag(self):
        """Coroutine methods are flagged async, plain ones are not."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_async_model"].is_async and not specs["demo_default_arg"].is_async


class TestQueryModel:
    def test_decode_query_applies_defaults(self):
        """Omitted parameters with defaults resolve to their default values."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].serializer.decode_query({"a": 5}) == {"a": 5, "b": 10}

    def test_decode_query_parses_nested_model(self):
        """Nested pydantic payloads are revived into real model instances."""
        specs = collect_rpc_method_specs(_GoodWorker)
        kwargs = specs["demo_async_model"].serializer.decode_query({"payload": {"text": "hi"}})
        assert kwargs["payload"] == _Payload(text="hi")

    def test_missing_required_param_rejected(self):
        """Missing required parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({})

    def test_unknown_param_rejected(self):
        """Extra unknown parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": 1, "unknown": 2})

    def test_wrong_type_rejected(self):
        """Type-mismatched parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": "not-an-int"})


class TestPostponedAnnotations:
    def test_string_annotations_resolved_in_worker_module(self):
        """A worker module using postponed annotations still builds real typed models."""
        specs = collect_rpc_method_specs(PostponedWorker)
        kwargs = specs["demo_transform"].serializer.decode_query({"payload": {"text": "hi"}})
        assert kwargs["payload"] == LatePayload(text="hi")

    def test_string_return_annotation_resolved(self):
        """A postponed return annotation resolves into a working result adapter."""
        specs = collect_rpc_method_specs(PostponedWorker)
        assert specs["demo_transform"].serializer.decode_result({"text": "hi"}) == LatePayload(text="hi")


class TestInheritance:
    def test_inherited_methods_collected(self):
        """Methods inherited from a base worker class are exposed too."""

        class Child(_GoodWorker):
            def demo_child_only(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Child)
        assert {"demo_default_arg", "demo_async_model", "demo_grouped", "demo_child_only"} <= set(specs)


class TestResultAdapter:
    def test_result_roundtrip(self):
        """Return values survive a json dump + validate roundtrip."""
        specs = collect_rpc_method_specs(_GoodWorker)
        dumped = specs["demo_async_model"].serializer.encode_result(_Payload(text="hi"))
        assert specs["demo_async_model"].serializer.decode_result(dumped) == _Payload(text="hi")

    def test_none_return_annotation(self):
        """Methods annotated -> None get a NoneType result adapter."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_grouped"].serializer.decode_result(None) is None


class TestFailLoud:
    def test_async_method_with_non_default_concurrency_group_rejected(self):
        """An async method with a non-default concurrency group fails at collection time."""

        class Worker:
            @rpc(concurrency_group="train")
            async def demo_async_grouped(self) -> int:
                return 0

        with pytest.raises(TypeError, match="concurrency_group"):
            collect_rpc_method_specs(Worker)

    def test_missing_param_annotation_rejected(self):
        """A parameter without a type annotation fails at collection time."""

        class Worker:
            def demo_unannotated_arg(self, x) -> int:
                return 0

        with pytest.raises(TypeError, match="must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_missing_return_annotation_rejected(self):
        """A method without a return annotation fails at collection time."""

        class Worker:
            def demo_unannotated_return(self, x: int):
                return x

        with pytest.raises(TypeError, match="return type annotation"):
            collect_rpc_method_specs(Worker)

    def test_var_positional_rejected(self):
        """*args signatures fail at collection time."""

        class Worker:
            def demo_var_positional(self, *x: int) -> int:
                return 0

        with pytest.raises(TypeError, match="args"):
            collect_rpc_method_specs(Worker)

    def test_var_keyword_rejected(self):
        """**kwargs signatures fail at collection time."""

        class Worker:
            def demo_var_keyword(self, **x: int) -> int:
                return 0

        with pytest.raises(TypeError, match="kwargs"):
            collect_rpc_method_specs(Worker)

    def test_positional_only_rejected(self):
        """Positional-only parameters fail at collection time since calls pass kwargs."""

        class Worker:
            def demo_positional_only(self, x: int, /) -> int:
                return x

        with pytest.raises(TypeError, match="positional-only"):
            collect_rpc_method_specs(Worker)

    def test_non_self_receiver_rejected(self):
        """An unconventionally named receiver is refused rather than silently dropped."""

        class Worker:
            def demo_odd_receiver(this, x: int) -> int:
                return x

        with pytest.raises(TypeError, match="receiver parameter 'self'"):
            collect_rpc_method_specs(Worker)

    def test_forgotten_self_is_rejected_instead_of_eating_the_first_argument(self):
        """A method that forgets self would otherwise lose its first parameter off the wire."""

        class Worker:
            def demo_forgot_self(a: int, b: int) -> int:
                return a + b

        with pytest.raises(TypeError, match="receiver parameter 'self'"):
            collect_rpc_method_specs(Worker)

    def test_wrapped_async_method_stays_async(self):
        """A functools.wraps-decorated async method is still detected as async."""

        def passthrough(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            return wrapper

        class Worker:
            @passthrough
            async def demo_wrapped_async(self, x: int) -> int:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_wrapped_async"].is_async
        assert specs["demo_wrapped_async"].serializer.decode_query({"x": 1}) == {"x": 1}

    def test_no_public_methods_rejected(self):
        """A worker class with no public methods fails at collection time."""

        class Worker:
            def _demo_hidden(self, x: int) -> int:
                return x

        with pytest.raises(TypeError, match="no public rpc methods"):
            collect_rpc_method_specs(Worker)

    def test_any_annotation_allowed(self):
        """Any-annotated parameters are accepted and passed through."""

        class Worker:
            def demo_any(self, x: Any) -> Any:
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_any"].serializer.decode_query({"x": [1, "a"]}) == {"x": [1, "a"]}
