"""Tinker-compatible wire layer over the multi-LoRA control plane.

Mount with ``--multi-lora-http-server-path
miles.ray.multi_lora.tinker.http_server.TinkerHTTPServer``; the base
control-plane routes stay available next to the ``/api/v1`` surface."""

from fastapi import FastAPI

from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer


class TinkerHTTPServer(MultiLoRAHTTPServer):
    """Serves the tinker SDK REST surface (tinker==0.24.1 wire contract)."""

    def add_routes(self, app: FastAPI) -> None:
        super().add_routes(app)
        app.get("/api/v1/get_server_capabilities")(self.get_server_capabilities)

    async def get_server_capabilities(self) -> dict:
        # One trainer serves one base model; adapters register on top of it.
        return {"supported_models": [{"model_name": self.backend.args.hf_checkpoint}]}
