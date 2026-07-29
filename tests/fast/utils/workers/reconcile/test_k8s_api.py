from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.reconcile.k8s_api import PodWatchEvent, exception_rejects_cursor


def make_exception(**fields: Any) -> Exception:
    exception = Exception("boom")
    for name, value in fields.items():
        setattr(exception, name, value)
    return exception


class TestResourceVersionParsing:
    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (SimpleNamespace(metadata=SimpleNamespace(resource_version="7")), "7"),
            (dict(metadata=dict(resourceVersion="7")), "7"),
            (dict(metadata=dict(resource_version="7")), "7"),
        ],
    )
    def test_both_wire_shapes_and_both_spellings_are_read(self, obj: Any, expected: str) -> None:
        """The client hands back deserialized models or raw dicts, camelCase or snake_case."""
        assert PodWatchEvent.from_frame(event_type="ADDED", obj=obj).resource_version == expected

    @pytest.mark.parametrize(
        "obj",
        [
            SimpleNamespace(),
            SimpleNamespace(metadata=None),
            SimpleNamespace(metadata=SimpleNamespace()),
            {},
            dict(metadata=None),
            dict(metadata={}),
            "a payload that is not an object at all",
        ],
    )
    def test_a_frame_without_a_readable_version_parses_to_none(self, obj: Any) -> None:
        """A missing or malformed metadata block must parse to None, never raise: the caller keeps its cursor."""
        assert PodWatchEvent.from_frame(event_type="ADDED", obj=obj).resource_version is None


class TestCursorRejection:
    @pytest.mark.parametrize(
        "obj",
        [
            SimpleNamespace(code=410, reason="Expired"),
            SimpleNamespace(code=504, reason="ResourceVersionTooLarge"),
            SimpleNamespace(code=None, reason="Expired"),
            dict(code=410),
            dict(reason="ResourceVersionTooLarge"),
        ],
    )
    def test_an_error_frame_reporting_a_dead_cursor_is_flagged(self, obj: Any) -> None:
        """Either the code or the reason is enough, in either wire shape."""
        assert PodWatchEvent.from_frame(event_type="ERROR", obj=obj).rejects_cursor

    @pytest.mark.parametrize(
        ("event_type", "obj"),
        [
            ("ERROR", SimpleNamespace(code=500, reason="InternalError")),
            ("ERROR", dict(code=500)),
            ("ERROR", SimpleNamespace()),
            ("MODIFIED", SimpleNamespace(code=410, reason="Expired")),
            ("BOOKMARK", dict(code=410)),
        ],
    )
    def test_anything_else_leaves_the_cursor_alone(self, event_type: str, obj: Any) -> None:
        """Only an ERROR frame may invalidate a cursor, and only for a cursor-specific code."""
        assert not PodWatchEvent.from_frame(event_type=event_type, obj=obj).rejects_cursor


class TestExceptionRejection:
    @pytest.mark.parametrize(
        "exception",
        [
            make_exception(status=410),
            make_exception(status="410"),
            make_exception(status=504),
            make_exception(code=410),
        ],
    )
    def test_a_client_exception_reporting_a_dead_cursor_is_flagged(self, exception: BaseException) -> None:
        """The status arrives as an int or a string, and some failures carry only a code."""
        assert exception_rejects_cursor(exception)

    @pytest.mark.parametrize(
        "exception",
        [make_exception(), make_exception(status=500), make_exception(status="500"), make_exception(code=500)],
    )
    def test_any_other_exception_is_a_plain_stream_failure(self, exception: BaseException) -> None:
        """A transient error must be retried on the same cursor instead of forcing a relist."""
        assert not exception_rejects_cursor(exception)
