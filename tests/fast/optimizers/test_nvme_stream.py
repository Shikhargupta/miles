"""The file-backed buffers behind Muon's disk-resident optimizer state.

Guards a silent failure: if the allocator stops returning file-backed storage, the offloader
keeps working against pinned host memory while the log still claims otherwise.
"""

import os

import pytest
import torch

from miles_plugins.optimizers import nvme_stream


def test_disk_buffer_matches_shape_and_dtype_and_is_not_pinned(tmp_path):
    src = torch.randn(64, 32, dtype=torch.float32)

    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    assert buf.shape == src.shape
    assert buf.dtype == src.dtype
    assert buf.device.type == "cpu"
    # The inherited offloader picks its sync/async copy path off is_pinned().
    assert not buf.is_pinned()


def test_disk_buffer_round_trip_is_bit_exact(tmp_path):
    src = torch.randn(128, 64, dtype=torch.float32)
    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    buf.copy_(src)
    out = torch.empty_like(src)
    out.copy_(buf)

    assert torch.equal(out, src)


def test_disk_buffer_leaves_no_file_behind(tmp_path):
    nvme_stream._disk_backed_like(torch.zeros(8), str(tmp_path))

    # Unlinked at creation, so a killed run leaves no residue.
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disk_buffer_preserves_dtype(tmp_path, dtype):
    src = torch.zeros(16, 4, dtype=dtype)

    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    assert buf.dtype is dtype
    assert buf.numel() == src.numel()
