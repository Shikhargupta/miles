"""Sampling transport for the tinker frontend
(codex-rollout-fullparameter-design-0810 §4.6).

The sampling hot path stays frontend -> router: /asample answers with a
future immediately and a background task posts the generation itself. This
port isolates WHERE that post goes — the SGLang router today, whatever
endpoint the InferenceController advertises after PR #1842 — without ever
proxying per-sample traffic through a rollout component. Serving identity,
versions, and session invalidation stay in the tinker backend/frontend:
only the HTTP hop lives here."""

import asyncio
from typing import Protocol

import httpx


class SamplingTransport(Protocol):
    async def generate(self, payload: dict) -> dict: ...

    async def close(self) -> None: ...


class SGLangRouterSamplingTransport:
    """Direct router transport with an explicit hard bound on in-flight
    generations (lazy client creation on the first request, like before).

    The previous default-configured client carried an implicit
    ``max_connections=100`` pool with a 10-second pool timeout: above 100
    concurrent generations (2 SDK clients x 64, before ``num_samples``
    fan-out) request #101 died waiting for a connection — an empty-message
    ``PoolTimeout`` the frontend turned into a terminal server failure the
    SDK never retries (the Tau 100/28 sampling cliff). The bound here is the
    transport-level invariant behind the frontend's weighted admission: even
    a caller that bypasses admission cannot stampede the router."""

    def __init__(self, base_url: str, max_inflight: int = 64) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_inflight = max_inflight
        # Acquired INSIDE each per-sample generation task (not at submit):
        # `async with` guarantees a sibling-cancelled or shutdown-cancelled
        # generation releases its permit on the way out.
        self._gate = asyncio.Semaphore(max_inflight)
        # The pool matches the gate, and pool=None removes the 10s pool
        # deadline. That is safe ONLY because the semaphore keeps in-flight
        # requests <= max_connections, so a request never actually queues on
        # the pool: legal, bounded waiting happens on the gate instead of
        # being misclassified as a terminal PoolTimeout. Read stays at 600s
        # (the value this frontend always used) — deriving it from router
        # config is deliberately out of scope here.
        self.limits = httpx.Limits(max_connections=max_inflight, max_keepalive_connections=max_inflight)
        self.timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=None)
        self._http: httpx.AsyncClient | None = None

    async def generate(self, payload: dict) -> dict:
        async with self._gate:
            if self._http is None:
                self._http = httpx.AsyncClient(limits=self.limits, timeout=self.timeout)
            response = await self._http.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
