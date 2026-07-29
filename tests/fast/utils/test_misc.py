import logging
import os

import pytest

from miles.utils.misc import FunctionRegistry, NodeProbeMixin, filter_keys, function_registry, load_function


def _fn_a():
    return "a"


def _fn_b():
    return "b"


class TestFunctionRegistry:
    def test_register_and_get(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a

    def test_register_duplicate_raises(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            with pytest.raises(AssertionError):
                with registry.temporary("my_fn", _fn_b):
                    pass

    def test_unregister(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a
        assert registry.get("my_fn") is None

    def test_temporary_cleanup_on_exception(self):
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", _fn_a):
                raise RuntimeError("test")
        assert registry.get("temp_fn") is None


class TestLoadFunction:
    def test_load_from_module(self):
        import os.path

        assert load_function("os.path.join") is os.path.join

    def test_load_none_returns_none(self):
        assert load_function(None) is None

    def test_load_from_registry(self):
        with function_registry.temporary("test:my_fn", _fn_a):
            assert load_function("test:my_fn") is _fn_a

    def test_registry_takes_precedence(self):
        with function_registry.temporary("os.path.join", _fn_b):
            assert load_function("os.path.join") is _fn_b
        assert load_function("os.path.join") is os.path.join


class TestFilterKeys:
    def test_projects_dict_by_keys(self):
        """filter_keys returns only the requested keys with their values."""
        d = {"a": 1, "b": 2, "c": 3}
        assert filter_keys(d, ["a", "c"]) == {"a": 1, "c": 3}

    def test_empty_interest_keys_returns_empty_dict(self):
        """An empty interest list yields an empty dict regardless of input."""
        assert filter_keys({"a": 1, "b": 2}, []) == {}

    def test_preserves_interest_keys_order(self):
        """Result key order follows interest_keys, not the source dict order."""
        d = {"a": 1, "b": 2, "c": 3}
        assert list(filter_keys(d, ["c", "a"]).keys()) == ["c", "a"]

    def test_full_subset_returns_all_entries(self):
        """Requesting every key returns the whole projection."""
        d = {"x": 10, "y": 20}
        assert filter_keys(d, ["x", "y"]) == {"x": 10, "y": 20}

    def test_duplicate_interest_key_collapses_to_single_entry(self):
        """A repeated interest key produces a single dict entry."""
        d = {"a": 1, "b": 2}
        assert filter_keys(d, ["a", "a"]) == {"a": 1}

    def test_missing_key_raises_key_error_and_logs(self, caplog):
        """A missing key raises KeyError and logs the error with context."""
        d = {"a": 1}
        with caplog.at_level(logging.ERROR, logger="miles.utils.misc"):
            with pytest.raises(KeyError):
                filter_keys(d, ["a", "missing"])
        assert any("filter_keys" in record.message for record in caplog.records)


class TestNodeProbeMixin:
    def test_get_node_ip_returns_nonempty_string(self):
        """The node ip probe answers with a usable address string."""
        node_ip = NodeProbeMixin._get_node_ip()
        assert isinstance(node_ip, str) and node_ip

    def test_get_free_port_block_returns_first_port_at_or_above_start(self):
        """A block request returns the first port of a free consecutive range."""
        first_port = NodeProbeMixin._get_free_port_block(start_port=15000, count=2)
        assert first_port >= 15000

    def test_get_gpu_uuids_returns_one_entry_per_gpu(self):
        """The uuid probe is best-effort: without NVML it still answers per gpu."""
        uuids = NodeProbeMixin._get_gpu_uuids([0, 1, 2])
        assert len(uuids) == 3
        assert all(uuid is None or isinstance(uuid, str) for uuid in uuids)
