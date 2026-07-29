import time

import pytest
import requests


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


class _FakeResponse:
    def __init__(self, status_code=202, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _registry_args():
    # Registry-API router: version > 0.2.1 and miles router disabled.
    return type("Args", (), {"use_miles_router": False})()


def test_register_worker_waits_until_observable(monkeypatch):
    """202 Accepted is the START of registration: the helper must poll GET /workers
    until the worker URL appears, not return on acceptance."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils import sglang_engine as se

    monkeypatch.setattr(se.sglang_router, "__version__", "0.3.1")
    monkeypatch.setattr(se.time, "sleep", lambda s: None)
    worker_url = "http://w:1"

    monkeypatch.setattr(se.requests, "post", lambda *a, **k: _FakeResponse(status_code=202))
    # Absent on the first two polls, then observable.
    listings = [[], [], [{"url": worker_url, "id": "uuid-1"}]]
    monkeypatch.setattr(se.requests, "get", lambda *a, **k: _FakeResponse(payload={"workers": listings.pop(0)}))

    se._register_worker_with_router(
        router_base="http://r:9",
        worker_url=worker_url,
        worker_type="regular",
        bootstrap_port=None,
        args=_registry_args(),
    )
    assert listings == []  # consumed all three listings, i.e. polled until observable


def test_register_worker_fast_fails_on_definitive_rejection(monkeypatch):
    """A definitive 4xx (not 408/429) must fail immediately, never poll."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils import sglang_engine as se

    monkeypatch.setattr(se.sglang_router, "__version__", "0.3.1")
    monkeypatch.setattr(se.requests, "post", lambda *a, **k: _FakeResponse(status_code=400))

    def _no_poll(*a, **k):
        raise AssertionError("must not poll the registry after a definitive rejection")

    monkeypatch.setattr(se.requests, "get", _no_poll)
    with pytest.raises(RuntimeError, match="rejected worker registration"):
        se._register_worker_with_router(
            router_base="http://r:9",
            worker_url="http://w:1",
            worker_type="regular",
            bootstrap_port=None,
            args=_registry_args(),
        )


def test_remove_worker_waits_until_absent(monkeypatch):
    """Removal resolves the UUID from the registry, deletes, then polls until the URL is gone."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils import sglang_engine as se

    monkeypatch.setattr(se.sglang_router, "__version__", "0.3.1")
    monkeypatch.setattr(se.time, "sleep", lambda s: None)
    worker_url = "http://w:1"

    # First listing resolves the id (+ the first absence poll), then it's gone.
    listings = [[{"url": worker_url, "id": "uuid-1"}], [{"url": worker_url, "id": "uuid-1"}], []]
    monkeypatch.setattr(se.requests, "get", lambda *a, **k: _FakeResponse(payload={"workers": listings.pop(0)}))
    deleted = []
    monkeypatch.setattr(
        se.requests, "delete", lambda url, **k: (deleted.append(url), _FakeResponse(status_code=200))[1]
    )

    se._remove_worker_from_router(router_base="http://r:9", worker_url=worker_url, args=_registry_args())
    assert deleted == ["http://r:9/workers/uuid-1"]
    assert listings == []  # polled until the worker disappeared
