"""Manager-level data source for multi-LoRA (Option 1).

The real per-adapter data sources are owned by the ``MultiLoRARolloutFn``
wrapper (one per registration, stamping serving identity at sample time); the
RolloutManager-level source is a no-op facade satisfying the DataSource
contract. The snapshot helpers here are shared with the rollout-layer scoped
aborts."""

import logging
from argparse import Namespace

import ray

from miles.ray.multi_lora.controller import get_multi_lora_controller
from miles.rollout.data_source import DataSource
from miles.utils.adapter_config import AdapterRun
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def fetch_snapshot() -> dict:
    return ray.get(get_multi_lora_controller().snapshot.remote())


def sampleable(snapshot: dict) -> dict[str, AdapterRun]:
    return {**snapshot["active"], **snapshot["retiring"]}


class MultiLoRANullDataSource(DataSource):
    """No-op DataSource: the Option 1 wrapper owns one real RolloutDataSource
    per adapter registration, so there is nothing to sample, save, or load at
    the manager level."""

    def __init__(self, args: Namespace):
        self.args = args

    def get_samples(self, num_samples: int = 1) -> list[list[Sample]]:
        return []

    def add_samples(self, samples: list[list[Sample]]) -> None:
        pass

    def save(self, rollout_id) -> None:
        pass

    def load(self, rollout_id=None) -> None:
        pass
