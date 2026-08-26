"""Tinker-compatible wire layer over the multi-LoRA control plane.

Mount with ``--multi-lora-http-server-path
miles.ray.multi_lora.tinker.http_server.TinkerHTTPServer``; the base
control-plane routes stay available next to the ``/api/v1`` surface."""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer
from miles.utils.adapter_config import AdapterRunConfig


class TinkerHTTPServer(MultiLoRAHTTPServer):
    """Serves the tinker SDK REST surface (tinker==0.24.1 wire contract)."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        super().__init__(backend, host, api_port)
        self._sessions: dict[str, dict] = {}
        self._models: dict[str, dict] = {}  # model_id -> {"name", "rank"}
        self._ready_futures: dict[str, dict] = {}  # request_id -> terminal body

    def add_routes(self, app: FastAPI) -> None:
        super().add_routes(app)
        app.get("/api/v1/get_server_capabilities")(self.get_server_capabilities)
        app.post("/api/v1/client/config")(self.client_config)
        app.post("/api/v1/create_session")(self.create_session)
        app.post("/api/v1/session_heartbeat")(self.session_heartbeat)
        app.post("/api/v1/telemetry")(self.telemetry)
        app.post("/api/v1/create_model")(self.create_model)
        app.post("/api/v1/get_info")(self.get_info)
        app.post("/api/v1/retrieve_future")(self.retrieve_future)

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

    # ------------------------------ model lifecycle ------------------------------

    async def create_model(self, request: Request):
        body = await request.json()
        session_id = body.get("session_id")
        if session_id not in self._sessions:
            return JSONResponse({"detail": f"unknown session '{session_id}'"}, status_code=404)
        model_id = f"{session_id}:train:{body['model_seq_id']}"
        request_id = f"{model_id}:create"
        if model_id not in self._models:  # an SDK retry replays the same ack
            name = f"tinker-{session_id.removeprefix('sess-')[:8]}-t{body['model_seq_id']}"
            lora = body.get("lora_config") or {}
            await self.backend.register(name, AdapterRunConfig(data="", rank=lora.get("rank"), alpha=lora.get("alpha")))
            self._models[model_id] = {"name": name, "rank": lora.get("rank")}
            self._ready_futures[request_id] = {"type": "create_model", "model_id": model_id}
        return {"request_id": request_id, "model_id": model_id}

    async def get_info(self, request: Request):
        model_id = (await request.json()).get("model_id")
        model = self._models.get(model_id)
        if model is None:
            return JSONResponse({"detail": f"unknown model '{model_id}'"}, status_code=404)
        base = self.backend.args.hf_checkpoint
        # arch/tokenizer_id are opaque strings to the SDK; the base checkpoint names both.
        return {
            "type": "get_info",
            "model_id": model_id,
            "model_name": base,
            "model_data": {"arch": base, "model_name": base, "tokenizer_id": base},
            "is_lora": True,
            "lora_rank": model["rank"],
        }

    async def retrieve_future(self, request: Request):
        request_id = (await request.json()).get("request_id")
        result = self._ready_futures.get(request_id)
        if result is None:
            # 410 marks a broken/unknown promise; the SDK treats it as retryable, never fatal.
            return JSONResponse({"detail": f"no result for request '{request_id}'"}, status_code=410)
        return result
