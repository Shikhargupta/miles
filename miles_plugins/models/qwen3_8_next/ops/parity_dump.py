"""Optional tensor dumps for the sglang forward-parity comparison.

Off unless the sglang dumper is enabled (``DUMPER_ENABLE``, or a POST to
``/dumper/configure``), and a no-op if the dumper is not importable at all, so the
training path neither changes behaviour nor gains a hard dependency. ``dumper.dump``
already returns immediately when disabled, so there is nothing to gate on here.

Names must match what the sglang side dumps at the same point -- the comparator
groups dump files into bundles by name, so a mismatched name silently drops the
comparison rather than failing it.
"""

from __future__ import annotations

from torch import Tensor


def parity_dump(name: str, tensor: Tensor, dims: str = "t 1 h # tp:replicated") -> None:
    try:
        from sglang.srt.debug_utils.dumper import dumper
    except Exception:
        return
    dumper.dump(name, tensor, dims=dims)
