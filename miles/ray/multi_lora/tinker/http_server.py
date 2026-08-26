"""Tinker-compatible wire layer over the multi-LoRA control plane.

Mount with ``--multi-lora-http-server-path
miles.ray.multi_lora.tinker.http_server.TinkerHTTPServer``; the base
control-plane routes stay available next to the ``/api/v1`` surface."""

import time
import uuid

from fastapi import FastAPI, Request

from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer


class TinkerHTTPServer(MultiLoRAHTTPServer):
    """Serves the tinker SDK REST surface (tinker==0.24.1 wire contract)."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        super().__init__(backend, host, api_port)
        self._sessions: dict[str, dict] = {}

    def add_routes(self, app: FastAPI) -> None:
        super().add_routes(app)
        app.get("/api/v1/get_server_capabilities")(self.get_server_capabilities)
        app.post("/api/v1/client/config")(self.client_config)
        app.post("/api/v1/create_session")(self.create_session)
        app.post("/api/v1/session_heartbeat")(self.session_heartbeat)
        app.post("/api/v1/telemetry")(self.telemetry)

    async def get_server_capabilities(self) -> dict:
        # One trainer serves one base model; adapters register on top of it.
        return {"supported_models": [{"model_name": self.backend.args.hf_checkpoint}]}

    # ------------------------------ session bootstrap ------------------------------

    async def client_config(self, request: Request) -> dict:
        # Empty response selects every SDK default: api-key auth, JSON wire, parallel chunks.
        return {}

    async def create_session(self, request: Request) -> dict:
        session_id = f"sess-{uuid.uuid4().hex[:16]}"
        self._sessions[session_id] = {"last_heartbeat": time.monotonic()}
        return {"type": "create_session", "session_id": session_id}

    async def session_heartbeat(self, request: Request) -> dict:
        session = self._sessions.get((await request.json()).get("session_id"))
        if session is not None:
            session["last_heartbeat"] = time.monotonic()
        return {"type": "session_heartbeat"}

    async def telemetry(self, request: Request) -> dict:
        return {"status": "accepted"}
