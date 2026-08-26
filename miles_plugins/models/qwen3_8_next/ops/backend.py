"""One switch per hand-written op family: torch reference vs triton kernel.

Every op we implement ourselves (QSA sparse attention, hyper-connection
mix/combine, PLE gate+conv) exists twice: a pure-torch reference that the
sglang parity runs verified, and a triton kernel set validated against that
reference (fp32 ~1e-7, bf16 at the rounding floor) by tests/qwen3_8_next/.
The reference stays the arbiter; the kernels are what training actually runs.

Selection is one env var per family, read per call so the parity harness can
flip backends inside one process:

    QSA_BACKEND / HC_BACKEND / PLE_BACKEND = "triton" | "torch"

Defaults live here in one place. A family flips to triton-by-default only
after an e2e A/B shows identical metrics.
"""

import os

_DEFAULTS = {
    "QSA": "triton",  # enabled in e2e since attempt 21; parity-tested fwd+bwd
    "HC": "torch",  # triton validated offline; flip after an e2e A/B
    "PLE": "torch",  # triton validated offline; flip after an e2e A/B
}


def backend(family: str) -> str:
    assert family in _DEFAULTS, family
    return os.environ.get(f"{family}_BACKEND", _DEFAULTS[family])


def use_triton(family: str) -> bool:
    return backend(family) == "triton"
