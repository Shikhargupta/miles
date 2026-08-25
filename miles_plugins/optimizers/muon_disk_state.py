"""Disk-backed optimizer state for Muon.

``--stream-optimizer-state-to-disk`` binds to ``DistributedOptimizer``, which Muon never
uses. Megatron's ``ChunkedOptimizerStateOffloader`` already carries Muon's state and ties to
host memory in exactly one place, ``_new_cpu_buffer``, so this subclasses it and overrides
that allocator to return file-backed tensors. File-backed pages are reclaimable page cache
rather than anonymous memory the kernel cannot evict.
"""

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)

_counter = threading.local()


def _next_index() -> int:
    value = getattr(_counter, "value", 0)
    _counter.value = value + 1
    return value


def _disk_backed_like(tensor: torch.Tensor, directory: str) -> torch.Tensor:
    """A tensor with ``tensor``'s shape/dtype whose storage is a file in ``directory``."""
    nbytes = tensor.numel() * tensor.element_size()
    path = os.path.join(directory, f"state_{os.getpid()}_{_next_index():06d}.bin")
    storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=max(nbytes, 1))
    # Unlinked so a killed run leaves nothing behind; the footprint stays visible in df only.
    os.unlink(path)
    return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(storage).view(tensor.size())


def offloader_class():
    """``ChunkedOptimizerStateOffloader`` with file-backed instead of pinned host buffers."""
    from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import ChunkedOptimizerStateOffloader

    class DiskOptimizerStateOffloader(ChunkedOptimizerStateOffloader):
        state_dir: str = "/scratch/miles_optimizer_state"

        def _new_cpu_buffer(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            os.makedirs(self.state_dir, exist_ok=True)
            buffer = _disk_backed_like(tensor, self.state_dir)
            self._disk_bytes = getattr(self, "_disk_bytes", 0) + buffer.numel() * buffer.element_size()
            return buffer

        def step(self) -> None:  # type: ignore[override]
            super().step()
            logger.info(f"Muon disk state step: {getattr(self, '_disk_bytes', 0) / 1024**3:.2f} GB file-backed")

    return DiskOptimizerStateOffloader


_installed = False


def install(state_dir: str | None = None) -> None:
    """Route Muon's chunked optimizer-state offload to disk instead of pinned host memory."""
    global _installed
    if _installed:
        return

    from megatron.core.optimizer import optimizer as megatron_optimizer_module
    from megatron.core.optimizer.cpu_offloading import chunked_optimizer_state_offload as defining_module

    disk_cls = offloader_class()
    if state_dir:
        disk_cls.state_dir = state_dir
    # optimizer.py imported the name directly, so rebinding only the defining module is a no-op.
    megatron_optimizer_module.ChunkedOptimizerStateOffloader = disk_cls
    defining_module.ChunkedOptimizerStateOffloader = disk_cls
    logger.info(f"Muon optimizer state on disk: buffers backed by files under {disk_cls.state_dir}")
    _installed = True
