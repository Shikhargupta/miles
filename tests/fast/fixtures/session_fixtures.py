from __future__ import annotations

from typing import Any

from miles.rollout.session.config import SessionServerConfig


def make_session_server_config(**overrides: Any) -> SessionServerConfig:
    defaults: dict[str, Any] = dict(
        host="127.0.0.1",
        port=0,
        instance_id=None,
        backend_url="http://127.0.0.1:0",
        timeout=30,
        hf_checkpoint=None,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        tito_allowed_append_roles=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )
    defaults.update(overrides)
    return SessionServerConfig(**defaults)
