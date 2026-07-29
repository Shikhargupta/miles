# doc-dev: docs/ci/02-docker-build.md
"""Verified wheel release materialization for Docker builds."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_ASSET_SUFFIXES = (".whl", ".tar.gz")
_BUFFER_SIZE = 1024 * 1024
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_SHA256_RE = re.compile(r"sha256:([0-9a-fA-F]{64})")
_TARGET_ARCHES = frozenset({"amd64", "arm64"})


class WheelCacheError(RuntimeError):
    """Raised when a wheel release cannot be materialized safely."""


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    size: int
    sha256: str
    download_url: str


def _validate_release_ref(repository: str, tag: str) -> None:
    if not _REPOSITORY_RE.fullmatch(repository):
        raise WheelCacheError(f"invalid wheels repository: {repository!r}")
    if not tag or tag in {".", ".."} or "/" in tag or "\\" in tag or "\0" in tag:
        raise WheelCacheError(f"invalid wheels release tag: {tag!r}")


def _request(url: str, *, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": "miles-docker-build"}
    if accept is not None:
        headers["Accept"] = accept
    if url.startswith("https://api.github.com/") and (token := os.environ.get("GITHUB_TOKEN")):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _open(request: urllib.request.Request) -> BinaryIO:
    return urllib.request.urlopen(request)


def fetch_release_assets(repository: str, tag: str) -> tuple[ReleaseAsset, ...]:
    """Fetch and validate the wheel assets in one GitHub release."""
    _validate_release_ref(repository, tag)
    encoded_tag = urllib.parse.quote(tag, safe="")
    url = f"https://api.github.com/repos/{repository}/releases/tags/{encoded_tag}"
    with _open(_request(url, accept="application/vnd.github+json")) as response:
        payload = json.load(response)

    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise WheelCacheError(f"invalid release manifest for {repository}@{tag}")

    assets: list[ReleaseAsset] = []
    names: set[str] = set()
    for raw_asset in payload["assets"]:
        if not isinstance(raw_asset, dict):
            raise WheelCacheError(f"invalid asset entry in {repository}@{tag}")
        name = raw_asset.get("name")
        if not isinstance(name, str):
            raise WheelCacheError(f"asset without a valid name in {repository}@{tag}")
        if not name.endswith(_ASSET_SUFFIXES) or raw_asset.get("state") != "uploaded":
            continue
        if not name or name in {".", ".."} or "/" in name or "\\" in name or "\0" in name or Path(name).name != name:
            raise WheelCacheError(f"unsafe asset name in {repository}@{tag}: {name!r}")
        if name in names:
            raise WheelCacheError(f"duplicate asset name in {repository}@{tag}: {name}")

        size = raw_asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise WheelCacheError(f"invalid size for {repository}@{tag} asset {name}")

        digest = raw_asset.get("digest")
        digest_match = _SHA256_RE.fullmatch(digest) if isinstance(digest, str) else None
        if digest_match is None:
            raise WheelCacheError(f"missing SHA256 digest for {repository}@{tag} asset {name}")

        download_url = raw_asset.get("browser_download_url")
        parsed_url = urllib.parse.urlparse(download_url) if isinstance(download_url, str) else None
        if parsed_url is None or parsed_url.scheme != "https" or not parsed_url.netloc:
            raise WheelCacheError(f"invalid download URL for {repository}@{tag} asset {name}")

        names.add(name)
        assets.append(
            ReleaseAsset(
                name=name,
                size=size,
                sha256=digest_match.group(1).lower(),
                download_url=download_url,
            )
        )

    if not assets:
        raise WheelCacheError(f"release {repository}@{tag} has no uploaded wheel assets")
    return tuple(sorted(assets, key=lambda asset: asset.name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_asset(path: Path, asset: ReleaseAsset) -> bool:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != asset.size:
            return False
        return _sha256(path) == asset.sha256
    except OSError:
        return False


def _download_asset(asset: ReleaseAsset, target: Path) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".part",
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(file_descriptor, "wb") as output, _open(_request(asset.download_url)) as response:
            while chunk := response.read(_BUFFER_SIZE):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != asset.size or digest.hexdigest() != asset.sha256:
            raise WheelCacheError(
                f"downloaded asset failed SHA256 validation: {asset.name} "
                f"(expected {asset.size} bytes/{asset.sha256}, got {size} bytes/{digest.hexdigest()})"
            )
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _release_lock(cache_root: Path, repository: str, tag: str, target_arch: str) -> Iterator[None]:
    lock_root = cache_root / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(f"{repository}\0{tag}\0{target_arch}".encode()).hexdigest()
    with (lock_root / f"{lock_key}.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cache_assets(
    cache_directory: Path,
    assets: tuple[ReleaseAsset, ...],
) -> tuple[tuple[Path, ...], int, int]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    hits = 0
    downloads = 0
    for asset in assets:
        target = cache_directory / asset.name
        if _matches_asset(target, asset):
            hits += 1
        else:
            print(f"wheel cache: downloading {asset.name} ({asset.size} bytes)", flush=True)
            _download_asset(asset, target)
            downloads += 1
        paths.append(target)
    return tuple(paths), hits, downloads


def materialize_release(
    repository: str,
    tag: str,
    directory: Path,
) -> tuple[tuple[Path, ...], int, int]:
    """Download and verify one wheel release into a directory."""
    return _cache_assets(directory, fetch_release_assets(repository, tag))


@contextmanager
def prepare_wheel_context(
    repository: str,
    releases: Mapping[str, str],
    cache_root: Path,
) -> Iterator[Path]:
    """Yield an isolated, complete wheel snapshot for one Docker build."""
    if not releases:
        raise WheelCacheError("no wheel releases configured for this build")
    unknown_arches = set(releases) - _TARGET_ARCHES
    if unknown_arches:
        raise WheelCacheError(f"unsupported wheel target architectures: {sorted(unknown_arches)}")

    cache_root = cache_root.expanduser()
    if not cache_root.is_absolute():
        raise WheelCacheError(f"MILES_DOCKER_CACHE_DIR must be absolute: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    snapshot_root = cache_root / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    total_hits = 0
    total_downloads = 0
    with tempfile.TemporaryDirectory(dir=snapshot_root, prefix="wheels-") as temporary_directory:
        context = Path(temporary_directory)
        for target_arch, tag in sorted(releases.items()):
            _validate_release_ref(repository, tag)
            with _release_lock(cache_root, repository, tag, target_arch):
                owner, name = repository.split("/", maxsplit=1)
                cache_directory = cache_root / "wheels" / owner / name / tag / target_arch
                cached_paths, hits, downloads = materialize_release(repository, tag, cache_directory)

                architecture_context = context / target_arch
                architecture_context.mkdir()
                for cached_path in cached_paths:
                    shutil.copy2(cached_path, architecture_context / cached_path.name)

                total_hits += hits
                total_downloads += downloads

        print(
            f"wheel cache: {total_hits} hit(s), {total_downloads} download(s); snapshot ready at {context}",
            flush=True,
        )
        yield context


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify one wheel release.")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, hits, downloads = materialize_release(args.repository, args.tag, args.output)
    print(f"wheel cache: {hits} hit(s), {downloads} download(s)", flush=True)


if __name__ == "__main__":
    main()
