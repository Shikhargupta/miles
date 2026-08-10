"""The environment variables a pod's processes agree on.

Kept free of imports: the outer serve entrypoint reads them before it is allowed to
pull in anything heavy, and the supervisor that sets them lives on top of torch.
"""

from __future__ import annotations

SUBPROCESS_INDEX_ENV_VAR = "MILES_SUPERVISOR_SUBPROCESS_INDEX"
CELL_INDEX_ENV_VAR = "MILES_CELL_INDEX"
POD_INDEX_ENV_VAR = "MILES_POD_INDEX"

PLATFORM_IDENTITY_ENV_VARS = (SUBPROCESS_INDEX_ENV_VAR, CELL_INDEX_ENV_VAR, POD_INDEX_ENV_VAR)
