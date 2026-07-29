#!/usr/bin/env python3
# doc-dev: docs/ci/02-docker-build.md
"""Build and push Miles Docker images.

Usage:
    python docker/build.py --variant cu13 --image-tag dev --push          # multi-arch (amd64+arm64)
    python docker/build.py --variant cu13-x86 --image-tag dev --push      # single arch
    python docker/build.py --variant cu12-x86 --image-tag latest
    python docker/build.py --variant cu13 --image-tag dev --dry-run
"""

import os
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import typer

from wheel_cache import prepare_wheel_context

DEFAULT_CACHE_DIR = Path("/tmp/miles-docker-cache")
REPO_ROOT = Path(__file__).resolve().parent.parent
WHEELS_REPOSITORY = "yueming-yuan/miles-wheels"

VARIANTS = {
    "cu13": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64", "linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
        "wheel_releases": {
            "amd64": "cu130-x86_64",
            "arm64": "cu130-aarch64",
        },
    },
    "cu13-x86": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64"],
        "tag_postfix": "",
        "build_args": {},
        "wheel_releases": {"amd64": "cu130-x86_64"},
    },
    "cu13-aarch64": {
        "image": "radixark/miles",
        "platforms": ["linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
        "wheel_releases": {"arm64": "cu130-aarch64"},
    },
    "cu12-x86": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64"],
        "tag_postfix": "-cu12",
        "build_args": {
            "ENABLE_CUDA_13": "0",
            "SGLANG_IMAGE_TAG": "v0.5.16-cu129",
        },
        "wheel_releases": {"amd64": "cu129-x86_64"},
    },
    "rocm700-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm700-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "rocm/sgl-dev",
            "SGLANG_IMAGE_TAG": "v0.5.14-rocm700-mi35x-20260627",
            "SGLANG_USE_ROCM700A": "1",
        },
    },
    "rocm700-mi30x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm700-mi30x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx942",
            "SGLANG_IMAGE_TAG": "v0.5.10-rocm700-mi30x",
            "SGLANG_USE_ROCM700A": "1",
        },
    },
    "rocm720-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm720-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "rocm/sgl-dev",
            "SGLANG_IMAGE_TAG": "v0.5.14-rocm720-mi35x-20260627",
            "APPLY_ROCR_VMMFIX": "1",
            "TE_USE_WHEEL": "1",
        },
    },
}


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _cache_root() -> Path:
    cache_root = Path(os.environ.get("MILES_DOCKER_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()
    if not cache_root.is_absolute():
        raise typer.BadParameter(f"MILES_DOCKER_CACHE_DIR must be absolute: {cache_root}")
    return cache_root


def build_and_push(
    variant: str, image_tag: str, dry_run: bool, dockerfile: str, push: bool = False, custom_tag: str = ""
) -> None:
    config = VARIANTS[variant]
    # A variant may pin its own Dockerfile (e.g. ROCm); otherwise use the CLI default.
    dockerfile = config.get("dockerfile", dockerfile)
    image = config["image"]
    postfix = config.get("tag_postfix", "")
    platforms = config.get("platforms")
    wheel_releases = config.get("wheel_releases")

    if image_tag == "latest":
        tags = [f"{image}:latest{postfix}"]
    elif image_tag == "dev":
        prefix = config.get("tag_prefix", "dev")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        tags = [f"{image}:{prefix}{postfix}", f"{image}:{prefix}{postfix}-{timestamp}"]
    elif image_tag == "custom":
        if not custom_tag:
            raise typer.BadParameter("--custom-tag is required when --image-tag is custom")
        tags = [f"{image}:{custom_tag}{postfix}"]
    else:
        raise typer.BadParameter(f"Unknown image tag: {image_tag}")

    if wheel_releases and dry_run:
        wheel_context_manager = nullcontext(_cache_root() / "snapshots" / "DRY_RUN_CONTEXT")
    elif wheel_releases:
        wheel_context_manager = prepare_wheel_context(WHEELS_REPOSITORY, wheel_releases, _cache_root())
    else:
        wheel_context_manager = nullcontext(None)

    with wheel_context_manager as wheel_context:
        cmd = [
            "docker",
            "buildx",
            "build",
            "-f",
            dockerfile,
        ]

        if platforms:
            cmd += ["--platform", ",".join(platforms)]

        if wheel_context is not None:
            cmd += ["--build-context", f"wheel_assets={wheel_context}"]

        if push:
            cmd += ["--push"]

        # Proxy args (pass through if set in environment, check both cases)
        for arg_name in ["HTTP_PROXY", "HTTPS_PROXY"]:
            value = os.environ.get(arg_name.lower()) or os.environ.get(arg_name)
            if value:
                cmd += ["--build-arg", f"{arg_name}={value}"]

        cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

        # Variant-specific build args
        for key, value in config.get("build_args", {}).items():
            cmd += ["--build-arg", f"{key}={value}"]

        for tag in tags:
            cmd += ["-t", tag]

        # Context is repo root
        cmd += ["."]

        print(f"\n=== Building {' '.join(tags)} ===", flush=True)
        run(cmd, dry_run)


class Variant(str, Enum):
    cu13 = "cu13"
    cu13_x86 = "cu13-x86"
    cu13_aarch64 = "cu13-aarch64"
    cu12_x86 = "cu12-x86"
    rocm700_mi35x = "rocm700-mi35x"
    rocm700_mi30x = "rocm700-mi30x"
    rocm720_mi35x = "rocm720-mi35x"


class ImageTag(str, Enum):
    latest = "latest"
    dev = "dev"
    custom = "custom"


def main(
    variant: Variant = typer.Option(..., help="Build variant to use."),  # noqa: B008
    image_tag: ImageTag = typer.Option(..., help="Tag mode: latest, dev, or custom."),  # noqa: B008
    dockerfile: str = typer.Option("docker/Dockerfile", help="Path to the Dockerfile."),  # noqa: B008
    dry_run: bool = typer.Option(False, help="Print commands without executing them."),  # noqa: B008
    push: bool = typer.Option(False, help="Push images to registry after building."),  # noqa: B008
    custom_tag: str = typer.Option("", help="Custom tag name (required when --image-tag is custom)."),  # noqa: B008
) -> None:
    build_and_push(variant.value, image_tag.value, dry_run, dockerfile, push=push, custom_tag=custom_tag)


if __name__ == "__main__":
    typer.run(main)
