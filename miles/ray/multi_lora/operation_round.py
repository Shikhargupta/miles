"""Executes one round of client operations against the trainer; the async driver branches here under --tinker-mode."""

import logging

logger = logging.getLogger(__name__)

TINKER_HTTP_SERVER_PATH = "miles.ray.multi_lora.tinker.http_server.TinkerHTTPServer"
OPERATION_BACKEND_PATH = "miles.ray.multi_lora.operation_backend.MultiLoRAOperationBackend"


def apply_tinker_defaults(args):
    """Serve the tinker wire protocol unless the seams are explicitly overridden."""
    assert getattr(args, "tinker_mode", False), "operation rounds require --tinker-mode"
    args.multi_lora_http_server_path = args.multi_lora_http_server_path or TINKER_HTTP_SERVER_PATH
    args.multi_lora_backend_path = args.multi_lora_backend_path or OPERATION_BACKEND_PATH
    return args


async def run_operation_round(args, actor_model, rollout_id: int) -> int:
    """Claim and execute one round of client operations (data co-batch plus control RPCs)."""
    raise NotImplementedError("operation-round execution lands with the next slice")
