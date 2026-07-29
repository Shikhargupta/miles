import hashlib
import importlib.util
import io
import json
import os
import sys
import threading
from pathlib import Path

import pytest

DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
SPEC = importlib.util.spec_from_file_location("miles_wheel_cache", DOCKER_DIR / "wheel_cache.py")
assert SPEC is not None and SPEC.loader is not None
wheel_cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wheel_cache
SPEC.loader.exec_module(wheel_cache)


def _asset(content: bytes, name: str = "package.whl"):
    return wheel_cache.ReleaseAsset(
        name=name,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        download_url=f"https://github.com/example/wheels/releases/download/tag/{name}",
    )


def _manifest_asset(content: bytes, **overrides):
    asset = {
        "name": "package.whl",
        "state": "uploaded",
        "size": len(content),
        "digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
        "browser_download_url": "https://github.com/example/wheels/releases/download/tag/package.whl",
    }
    asset.update(overrides)
    return asset


def _response(payload) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def test_fetch_release_assets_validates_manifest(monkeypatch):
    content = b"wheel"
    monkeypatch.setattr(
        wheel_cache,
        "_open",
        lambda request: _response(
            {
                "assets": [
                    _manifest_asset(content),
                    _manifest_asset(b"ignored", name="notes.txt", digest=None),
                ]
            }
        ),
    )

    assets = wheel_cache.fetch_release_assets("example/wheels", "cu130-x86_64")

    assert assets == (_asset(content),)


@pytest.mark.parametrize(
    "asset, error",
    [
        (_manifest_asset(b"wheel", name="../package.whl"), "unsafe asset name"),
        (_manifest_asset(b"wheel", size=-1), "invalid size"),
        (_manifest_asset(b"wheel", digest=None), "missing SHA256 digest"),
        (_manifest_asset(b"wheel", browser_download_url="http://example.com/package.whl"), "invalid download URL"),
    ],
)
def test_fetch_release_assets_rejects_unsafe_manifest(monkeypatch, asset, error):
    monkeypatch.setattr(wheel_cache, "_open", lambda request: _response({"assets": [asset]}))

    with pytest.raises(wheel_cache.WheelCacheError, match=error):
        wheel_cache.fetch_release_assets("example/wheels", "cu130-x86_64")


@pytest.mark.parametrize(
    "assets, error",
    [
        ([], "has no uploaded wheel assets"),
        ([_manifest_asset(b"wheel"), _manifest_asset(b"wheel")], "duplicate asset name"),
    ],
)
def test_fetch_release_assets_rejects_empty_or_duplicate_assets(monkeypatch, assets, error):
    monkeypatch.setattr(wheel_cache, "_open", lambda request: _response({"assets": assets}))

    with pytest.raises(wheel_cache.WheelCacheError, match=error):
        wheel_cache.fetch_release_assets("example/wheels", "cu130-x86_64")


def test_matching_cached_asset_is_reused(tmp_path, monkeypatch):
    content = b"cached wheel"
    asset = _asset(content)
    target = tmp_path / asset.name
    target.write_bytes(content)
    monkeypatch.setattr(wheel_cache, "fetch_release_assets", lambda repository, tag: (asset,))
    monkeypatch.setattr(
        wheel_cache,
        "_download_asset",
        lambda asset, target: pytest.fail("matching cache entry must not be downloaded"),
    )

    paths, hits, downloads = wheel_cache.materialize_release("example/wheels", "tag", tmp_path)

    assert paths == (target,)
    assert (hits, downloads) == (1, 0)


def test_mismatched_cached_asset_is_atomically_replaced(tmp_path, monkeypatch):
    stale = b"stale wheel!"
    fresh = b"fresh wheel!"
    asset = _asset(fresh)
    target = tmp_path / asset.name
    target.write_bytes(stale)
    monkeypatch.setattr(wheel_cache, "_open", lambda request: io.BytesIO(fresh))

    real_replace = os.replace
    replacements = []

    def checked_replace(source, destination):
        source = Path(source)
        destination = Path(destination)
        assert source.parent == destination.parent
        assert hashlib.sha256(source.read_bytes()).hexdigest() == asset.sha256
        assert destination.read_bytes() == stale
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(wheel_cache.os, "replace", checked_replace)

    paths, hits, downloads = wheel_cache._cache_assets(tmp_path, (asset,))

    assert paths == (target,)
    assert (hits, downloads) == (0, 1)
    assert target.read_bytes() == fresh
    assert len(replacements) == 1
    assert list(tmp_path.glob("*.part")) == []


def test_bad_download_is_not_published(tmp_path, monkeypatch):
    stale = b"stale wheel"
    expected = b"fresh wheel"
    asset = _asset(expected)
    target = tmp_path / asset.name
    target.write_bytes(stale)
    monkeypatch.setattr(wheel_cache, "_open", lambda request: io.BytesIO(b"corrupt"))

    with pytest.raises(wheel_cache.WheelCacheError, match="failed SHA256 validation"):
        wheel_cache._cache_assets(tmp_path, (asset,))

    assert target.read_bytes() == stale
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize("fail_inside_context", [False, True])
def test_wheel_context_is_a_complete_physical_copy_and_is_always_cleaned(tmp_path, monkeypatch, fail_inside_context):
    repository = "example/wheels"
    releases = {
        "amd64": "x86-tag",
        "arm64": "arm-tag",
    }
    contents = {
        "x86-tag": b"x86 wheel",
        "arm-tag": b"arm wheel",
    }
    assets = {tag: _asset(content, f"{tag}.whl") for tag, content in contents.items()}
    for target_arch, tag in releases.items():
        cache_path = tmp_path / "wheels" / "example" / "wheels" / tag / target_arch / assets[tag].name
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(contents[tag])

    monkeypatch.setattr(wheel_cache, "fetch_release_assets", lambda repository, tag: (assets[tag],))
    monkeypatch.setattr(
        wheel_cache,
        "_download_asset",
        lambda asset, target: pytest.fail("matching cache entry must not be downloaded"),
    )

    context_path = None
    expected_error = RuntimeError("build failed")
    try:
        with wheel_cache.prepare_wheel_context(repository, releases, tmp_path) as context:
            context_path = context
            assert context.parent == tmp_path / "snapshots"
            assert sorted(path.relative_to(context) for path in context.rglob("*") if path.is_file()) == [
                Path("amd64/x86-tag.whl"),
                Path("arm64/arm-tag.whl"),
            ]
            for target_arch, tag in releases.items():
                cached = tmp_path / "wheels" / "example" / "wheels" / tag / target_arch / assets[tag].name
                snapshot = context / target_arch / assets[tag].name
                assert snapshot.read_bytes() == contents[tag]
                assert snapshot.stat().st_ino != cached.stat().st_ino
                cached.write_bytes(b"changed after snapshot")
                assert snapshot.read_bytes() == contents[tag]
            if fail_inside_context:
                raise expected_error
    except RuntimeError as error:
        assert fail_inside_context
        assert error is expected_error

    assert context_path is not None
    assert not context_path.exists()


def test_concurrent_contexts_serialize_cache_refresh(tmp_path, monkeypatch):
    content = b"shared wheel"
    asset = _asset(content)
    first_fetch_entered = threading.Event()
    release_first_fetch = threading.Event()
    second_fetch_entered = threading.Event()
    fetch_count = 0
    fetch_count_lock = threading.Lock()
    download_count = 0

    def fake_fetch(repository, tag):
        nonlocal fetch_count
        with fetch_count_lock:
            fetch_count += 1
            current_fetch = fetch_count
        if current_fetch == 1:
            first_fetch_entered.set()
            assert release_first_fetch.wait(timeout=5)
        else:
            second_fetch_entered.set()
        return (asset,)

    def fake_open(request):
        nonlocal download_count
        download_count += 1
        return io.BytesIO(content)

    monkeypatch.setattr(wheel_cache, "fetch_release_assets", fake_fetch)
    monkeypatch.setattr(wheel_cache, "_open", fake_open)

    def prepare(started):
        started.set()
        with wheel_cache.prepare_wheel_context(
            "example/wheels",
            {"amd64": "x86-tag"},
            tmp_path,
        ) as context:
            return (context / "amd64" / asset.name).read_bytes()

    first_started = threading.Event()
    second_started = threading.Event()
    first_result = {}
    second_result = {}

    first_thread = threading.Thread(target=lambda: first_result.setdefault("value", prepare(first_started)))
    second_thread = threading.Thread(target=lambda: second_result.setdefault("value", prepare(second_started)))
    first_thread.start()
    assert first_started.wait(timeout=5)
    assert first_fetch_entered.wait(timeout=5)
    second_thread.start()
    assert second_started.wait(timeout=5)
    assert not second_fetch_entered.wait(timeout=0.1)
    release_first_fetch.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first_result["value"] == content
    assert second_result["value"] == content
    assert fetch_count == 2
    assert download_count == 1
