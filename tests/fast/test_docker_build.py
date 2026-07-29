import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
sys.path.insert(0, str(DOCKER_DIR))
SPEC = importlib.util.spec_from_file_location("miles_docker_build", DOCKER_DIR / "build.py")
assert SPEC is not None and SPEC.loader is not None
docker_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docker_build)


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_cuda_dockerfile_has_self_contained_wheel_fallback():
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text()
    fallback_start = dockerfile.index("FROM sglang_base AS wheel_assets")
    final_start = dockerfile.index("FROM sglang_base AS sglang\n")
    fallback = dockerfile[fallback_start:final_start]

    assert fallback_start < final_start
    assert f"ARG WHEELS_REPO={docker_build.WHEELS_REPOSITORY}" in fallback
    assert f"ARG WHEELS_TAG_X86={docker_build.VARIANTS['cu13']['wheel_releases']['amd64']}" in fallback
    assert f"ARG WHEELS_TAG_ARM64={docker_build.VARIANTS['cu13']['wheel_releases']['arm64']}" in fallback
    assert "COPY docker/wheel_cache.py /tmp/wheel_cache.py" in fallback
    assert "python3 /tmp/wheel_cache.py" in fallback
    assert "COPY --from=wheel_assets /${TARGETARCH}/ /tmp/wheels/" in dockerfile[final_start:]


@pytest.mark.parametrize("build_fails", [False, True])
def test_cuda_build_uses_wheel_snapshot_for_the_full_build(tmp_path, monkeypatch, build_fails):
    cache_root = tmp_path / "cache"
    snapshot = cache_root / "snapshots" / "test-snapshot"
    observed = {}

    @contextmanager
    def fake_prepare(repository, releases, configured_cache_root):
        assert repository == "yueming-yuan/miles-wheels"
        assert releases == {
            "amd64": "cu130-x86_64",
            "arm64": "cu130-aarch64",
        }
        assert configured_cache_root == cache_root
        snapshot.mkdir(parents=True)
        try:
            yield snapshot
        finally:
            snapshot.rmdir()

    expected_error = RuntimeError("docker failed")

    def fake_run(command, dry_run):
        assert not dry_run
        assert snapshot.is_dir()
        observed["command"] = command
        if build_fails:
            raise expected_error

    monkeypatch.setenv("MILES_DOCKER_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(docker_build, "prepare_wheel_context", fake_prepare)
    monkeypatch.setattr(docker_build, "run", fake_run)

    if build_fails:
        with pytest.raises(RuntimeError) as raised:
            docker_build.build_and_push(
                "cu13",
                "custom",
                False,
                "docker/Dockerfile",
                push=True,
                custom_tag="test",
            )
        assert raised.value is expected_error
    else:
        docker_build.build_and_push(
            "cu13",
            "custom",
            False,
            "docker/Dockerfile",
            push=True,
            custom_tag="test",
        )

    assert not snapshot.exists()
    command = observed["command"]
    assert _option_value(command, "--platform") == "linux/amd64,linux/arm64"
    assert _option_value(command, "--build-context") == f"wheel_assets={snapshot}"
    assert "--push" in command


def test_cu12_build_uses_its_verified_wheel_snapshot(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    snapshot = cache_root / "snapshots" / "cu12-snapshot"
    observed = {}

    @contextmanager
    def fake_prepare(repository, releases, configured_cache_root):
        assert repository == "yueming-yuan/miles-wheels"
        assert releases == {"amd64": "cu129-x86_64"}
        assert configured_cache_root == cache_root
        snapshot.mkdir(parents=True)
        yield snapshot

    monkeypatch.setenv("MILES_DOCKER_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(docker_build, "prepare_wheel_context", fake_prepare)
    monkeypatch.setattr(
        docker_build,
        "run",
        lambda command, dry_run: observed.update(command=command, dry_run=dry_run),
    )

    docker_build.build_and_push(
        "cu12-x86",
        "custom",
        False,
        "docker/Dockerfile",
        custom_tag="test",
    )

    assert not observed["dry_run"]
    assert _option_value(observed["command"], "--build-context") == f"wheel_assets={snapshot}"
    assert "ENABLE_CUDA_13=0" in observed["command"]
    assert "SGLANG_IMAGE_TAG=v0.5.16-cu129" in observed["command"]


def test_rocm_build_does_not_prepare_cuda_wheels(monkeypatch):
    observed = {}
    monkeypatch.setattr(
        docker_build,
        "prepare_wheel_context",
        lambda *args, **kwargs: pytest.fail("ROCm build must not prepare CUDA wheels"),
    )
    monkeypatch.setattr(
        docker_build,
        "run",
        lambda command, dry_run: observed.update(command=command, dry_run=dry_run),
    )

    docker_build.build_and_push(
        "rocm700-mi35x",
        "custom",
        False,
        "docker/Dockerfile",
        custom_tag="test",
    )

    assert not observed["dry_run"]
    assert "--build-context" not in observed["command"]
    assert _option_value(observed["command"], "-f") == "docker/Dockerfile.rocm"


def test_cuda_dry_run_has_no_cache_io(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    observed = {}
    monkeypatch.setenv("MILES_DOCKER_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        docker_build,
        "prepare_wheel_context",
        lambda *args, **kwargs: pytest.fail("dry-run must not prepare wheels"),
    )
    monkeypatch.setattr(
        docker_build,
        "run",
        lambda command, dry_run: observed.update(command=command, dry_run=dry_run),
    )

    docker_build.build_and_push(
        "cu13-x86",
        "custom",
        True,
        "docker/Dockerfile",
        custom_tag="test",
    )

    assert observed["dry_run"]
    assert _option_value(observed["command"], "--build-context") == (
        f"wheel_assets={cache_root / 'snapshots' / 'DRY_RUN_CONTEXT'}"
    )
    assert not cache_root.exists()
