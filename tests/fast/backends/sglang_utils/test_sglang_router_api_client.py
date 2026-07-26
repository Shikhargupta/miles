import pytest
import requests

sglang_router = pytest.importorskip("sglang_router")

from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient  # noqa: E402

ROUTER_URL = "http://router-host:9000"
WORKER_URL = "http://fake-host:1234"


class _FakeResponse:
    def __init__(self, payload: dict | None = None):
        self._payload = payload if payload is not None else {"ok": True}
        self.raise_for_status_calls = 0

    def raise_for_status(self):
        self.raise_for_status_calls += 1

    def json(self):
        return self._payload


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.responses: list[_FakeResponse] = []

    def install(self, monkeypatch, responses: list[_FakeResponse] | None = None):
        self.responses = list(responses or [])
        for verb in ("get", "post", "delete"):
            monkeypatch.setattr(requests, verb, self._make_handler(verb))

    def _make_handler(self, verb: str):
        def handler(url, **kwargs):
            self.calls.append((verb, url, kwargs))
            if self.responses:
                return self.responses.pop(0)
            return _FakeResponse()

        return handler


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    return rec


@pytest.fixture
def client():
    return SGLangRouterApiClient(router_url=ROUTER_URL)


def test_add_worker_uses_the_query_string_api_when_legacy(client, recorder):
    """Routers <= 0.2.1 and the miles router only understand /add_worker?url=."""
    client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True)

    assert recorder.calls == [("post", f"{ROUTER_URL}/add_worker?url={WORKER_URL}", {})]


def test_add_worker_rejects_pd_disaggregation_on_the_legacy_api(client, recorder):
    """The legacy API has no worker_type concept, so prefill/decode workers must be refused."""
    with pytest.raises(AssertionError, match="pd disaggregation is not supported"):
        client.add_worker(worker_url=WORKER_URL, worker_type="prefill", use_legacy_api=True)


def test_add_worker_posts_the_worker_payload_on_the_modern_api(client, recorder):
    """Modern routers take a JSON body on /workers."""
    client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=False)

    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{ROUTER_URL}/workers")
    assert kwargs["json"] == {"url": WORKER_URL, "worker_type": "regular"}


def test_add_worker_includes_the_bootstrap_port_for_prefill_workers(client, recorder):
    """PD disaggregation needs the prefill worker's bootstrap port registered with the router."""
    client.add_worker(worker_url=WORKER_URL, worker_type="prefill", use_legacy_api=False, bootstrap_port=8998)

    assert recorder.calls[0][2]["json"] == {
        "url": WORKER_URL,
        "worker_type": "prefill",
        "bootstrap_port": 8998,
    }


def test_remove_worker_uses_the_query_string_api_when_legacy(client, recorder):
    """Legacy routers unregister via /remove_worker?url=."""
    client.remove_worker(worker_url=WORKER_URL, use_legacy_api=True)

    assert recorder.calls == [("post", f"{ROUTER_URL}/remove_worker?url={WORKER_URL}", {})]


def test_remove_worker_deletes_by_url_on_pre_0_3_routers(client, recorder, monkeypatch):
    """Routers in [0.2.2, 0.3.0) address workers by percent-encoded url."""
    monkeypatch.setattr(sglang_router, "__version__", "0.2.5")

    client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert recorder.calls == [("delete", f"{ROUTER_URL}/workers/http%3A%2F%2Ffake-host%3A1234", {})]


def test_remove_worker_resolves_the_worker_id_on_modern_routers(client, monkeypatch):
    """Routers >= 0.3.0 address workers by id, so the url must be resolved first."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-7"}]})])

    client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert [(verb, url) for verb, url, _kwargs in rec.calls] == [
        ("get", f"{ROUTER_URL}/workers"),
        ("delete", f"{ROUTER_URL}/workers/w-7"),
    ]


def test_remove_worker_tolerates_an_unknown_worker(client, monkeypatch):
    """Shutdown must not fail when the router no longer knows the worker."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": []})])

    client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert [verb for verb, _url, _kwargs in rec.calls] == ["get"]


def test_add_worker_propagates_router_errors(client, monkeypatch):
    """A router that rejects the registration must surface, not be swallowed."""
    rec = _Recorder()
    response = _FakeResponse()
    rec.install(monkeypatch, responses=[response])

    client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True)

    assert response.raise_for_status_calls == 1


def test_remove_worker_propagates_router_errors(client, monkeypatch):
    """Unregistration errors must surface on the legacy path too."""
    rec = _Recorder()
    response = _FakeResponse()
    rec.install(monkeypatch, responses=[response])

    client.remove_worker(worker_url=WORKER_URL, use_legacy_api=True)

    assert response.raise_for_status_calls == 1
