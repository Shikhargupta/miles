from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SessionServerConfig(FrozenStrictBaseModel):
    host: str
    port: int
    instance_id: str | None
    backend_url: str
    timeout: float | None
    hf_checkpoint: str | None
    chat_template_path: str | None
    tito_model: str
    apply_chat_template_kwargs: dict[str, Any] | None
    tito_allowed_append_roles: list[str] | None
    use_rollout_routing_replay: bool
    use_rollout_indexer_replay: bool


def compute_session_server_config(
    args, *, host: str, port: int, instance_id: str | None, backend_url: str
) -> SessionServerConfig:
    return SessionServerConfig(
        host=host,
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        timeout=args.miles_router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        tito_model=args.tito_model,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        tito_allowed_append_roles=args.tito_allowed_append_roles,
        use_rollout_routing_replay=args.use_rollout_routing_replay,
        use_rollout_indexer_replay=args.use_rollout_indexer_replay,
    )
