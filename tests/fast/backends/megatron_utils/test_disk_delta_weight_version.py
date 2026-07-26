from unittest import mock

import pytest

_DELTA = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta"
_RAY_GET = f"{_DELTA}.ray.get"


def _helpers():
    pytest.importorskip("sglang")
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import (
        _UNSET_WEIGHT_VERSION,
        _update_weight_version,
        _update_weight_version_if_unset,
    )

    return _UNSET_WEIGHT_VERSION, _update_weight_version, _update_weight_version_if_unset


class TestUpdateWeightVersion:
    def test_publishes_the_version_to_every_engine(self):
        """Paths whose transport cannot carry the version announce it to all engines itself."""
        _unset, update_weight_version, _if_unset = _helpers()
        calls = []

        class FakeEngine:
            def __init__(self, name):
                self.update_weight_version = _RecordingMethod(calls, name)

        with mock.patch(_RAY_GET, lambda refs: list(refs)):
            update_weight_version([FakeEngine("a"), FakeEngine("b")], "ab12cd34-00000007")

        assert calls == [("a", "ab12cd34-00000007", False), ("b", "ab12cd34-00000007", False)]

    def test_announcing_does_not_abort_running_requests(self):
        """Announcing a version changes no weights, so it must not cancel in-flight generation."""
        _unset, update_weight_version, _if_unset = _helpers()
        seen = {}

        class FakeEngine:
            class update_weight_version:
                @staticmethod
                def remote(weight_version, abort_all_requests):
                    seen["abort_all_requests"] = abort_all_requests
                    return object()

        with mock.patch(_RAY_GET, lambda refs: list(refs)):
            update_weight_version([FakeEngine()], "ab12cd34-00000007")

        assert seen["abort_all_requests"] is False

    def test_only_announces_to_engines_that_were_never_told_a_version(self):
        """A cold engine still reports the launch default and needs the version."""
        unset_version, _update, update_weight_version_if_unset = _helpers()
        announced = []

        class ColdEngine:
            def __init__(self):
                self.get_weight_version = _Returning(unset_version)
                self.update_weight_version = _Recording(announced)

        with mock.patch(_RAY_GET, _resolve):
            update_weight_version_if_unset([ColdEngine()], "ab12cd34-00000007")

        assert announced == ["ab12cd34-00000007"]

    def test_leaves_an_engine_that_already_serves_a_known_version_alone(self):
        """After a trainer failover the engines still hold their weights; claiming a version would lie."""
        _unset, _update, update_weight_version_if_unset = _helpers()
        announced = []

        class WarmEngine:
            def __init__(self):
                self.get_weight_version = _Returning("ab12cd34-00000004")
                self.update_weight_version = _Recording(announced)

        with mock.patch(_RAY_GET, _resolve):
            update_weight_version_if_unset([WarmEngine()], "ab12cd34-00000007")

        assert announced == []

    def test_announces_only_to_the_engines_that_are_still_cold(self):
        """Partial recovery leaves survivors holding real weights; only the replacements need telling."""
        unset_version, _update, update_weight_version_if_unset = _helpers()
        announced = []

        class Engine:
            def __init__(self, reported):
                self.get_weight_version = _Returning(reported)
                self.update_weight_version = _Recording(announced)

        warm = Engine("ab12cd34-00000004")
        cold = Engine(unset_version)
        never_told = Engine(None)

        with mock.patch(_RAY_GET, _resolve):
            update_weight_version_if_unset([warm, cold, never_told], "ab12cd34-00000007")

        assert announced == ["ab12cd34-00000007", "ab12cd34-00000007"]

    def test_no_engines_is_a_noop(self):
        """An updater with no engines attached publishes nothing rather than failing."""
        _unset, update_weight_version, _if_unset = _helpers()

        with mock.patch(_RAY_GET, lambda refs: list(refs)):
            update_weight_version([], "ab12cd34-00000007")


class _RecordingMethod:
    def __init__(self, calls, name):
        self._calls = calls
        self._name = name

    def remote(self, weight_version, abort_all_requests=True):
        self._calls.append((self._name, weight_version, abort_all_requests))
        return object()


class _Returning:
    def __init__(self, value):
        self._value = value

    def remote(self):
        return _Resolved(self._value)


class _Recording:
    def __init__(self, announced):
        self._announced = announced

    def remote(self, weight_version, abort_all_requests=True):
        self._announced.append(weight_version)
        return _Resolved(None)


class _Resolved:
    def __init__(self, value):
        self.value = value


def _resolve(refs):
    return [ref.value if isinstance(ref, _Resolved) else ref for ref in refs]
