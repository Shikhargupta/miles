"""torch API shims needed to import torchtitan in this image.

torchtitan tracks a newer torch than this image runs. Each shim covers a symbol
that torchtitan imports at module scope but only *uses* on a code path this
backend never takes, and each raises if it is actually exercised -- so a future
titan version that really needs one fails loudly instead of silently misbehaving.

The image is on torch 2.11.0. Moving off it is ours to schedule rather than
upstream's to deliver: sglang v0.5.18 already requires ``torch==2.13.0``, while
``docker/build.py`` pins the base image to v0.5.16 and the Dockerfile layers the
sglang-miles fork on it. Relaxing the version constraint alone would not work --
``sgl_kernel``'s extensions link libtorch directly, so the whole image moves or
nothing does.

Delete this module once the image is on torch>=2.12, which is where
``DataParallelMeshDims`` landed. Note that is *not* the same threshold as the
attention backends -- see validate_torchtitan_args.
"""

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    _shim_data_parallel_mesh_dims()


def _shim_data_parallel_mesh_dims() -> None:
    """``torch.distributed.fsdp.DataParallelMeshDims`` (torch 2.12+).

    torchtitan's ``distributed/full_dtensor`` imports it at module scope, and
    ``trainer`` -> ``validate`` -> ``full_dtensor`` drags that in on every path.
    It is only used to tell FSDP which mesh axes shard vs replicate under the
    ``full_dtensor`` / ``spmd_types`` SPMD backends, which this backend does not
    select.
    """
    import torch.distributed.fsdp as torch_fsdp

    if hasattr(torch_fsdp, "DataParallelMeshDims"):
        return

    class DataParallelMeshDims:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "DataParallelMeshDims requires torch>=2.12, but this image pins torch==2.11.0 "
                "(sglang requirement). It is only needed by torchtitan's full_dtensor / "
                "spmd_types SPMD backends; do not select those."
            )

    torch_fsdp.DataParallelMeshDims = DataParallelMeshDims
    logger.info("Installed DataParallelMeshDims compat shim for torchtitan on torch<2.12")
