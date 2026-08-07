from __future__ import annotations

from types import SimpleNamespace

from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer, compute_inference_weight
from miles.utils.context_lock import ContextLock


def _cell(*, num_gpus_per_engine: int, is_serving: bool):
    return SimpleNamespace(
        meta=SimpleNamespace(num_gpus_per_engine=num_gpus_per_engine),
        is_serving=is_serving,
    )


def _server(*, cells: dict, weight_per_cell: int | None) -> RolloutServer:
    return RolloutServer(
        server_cells=cells,
        args=make_args(inference_weight_per_cell=weight_per_cell),
        context_lock=ContextLock("RolloutServer"),
    )


class TestComputeInferenceWeight:
    def test_it_sums_gpu_counts_when_no_capacity_is_configured(self):
        """The default capacity of a cell is its own size, so a fleet's weight is its GPU count."""
        assert compute_inference_weight(serving_cell_gpu_counts=[4, 4, 8], weight_per_cell=None) == 16

    def test_it_uses_the_configured_capacity_per_cell_when_set(self):
        """An explicit capacity overrides the GPU-count heuristic for deployments it does not fit."""
        assert compute_inference_weight(serving_cell_gpu_counts=[4, 4, 8], weight_per_cell=10) == 30

    def test_an_empty_fleet_weighs_nothing(self):
        """A deployment with no serving cell must report zero, which drains it at the balancer."""
        assert compute_inference_weight(serving_cell_gpu_counts=[], weight_per_cell=None) == 0
        assert compute_inference_weight(serving_cell_gpu_counts=[], weight_per_cell=10) == 0

    def test_a_configured_capacity_of_zero_drains_the_deployment(self):
        """Zero is a legal capacity and must not silently fall back to the GPU count."""
        assert compute_inference_weight(serving_cell_gpu_counts=[4, 4], weight_per_cell=0) == 0


class TestRolloutServerInferenceWeight:
    def test_it_counts_only_serving_cells(self):
        """A cell that is still initializing carries no traffic, so counting it overstates capacity."""
        server = _server(
            cells={
                "a": _cell(num_gpus_per_engine=4, is_serving=True),
                "b": _cell(num_gpus_per_engine=4, is_serving=False),
                "c": _cell(num_gpus_per_engine=8, is_serving=True),
            },
            weight_per_cell=None,
        )
        assert server.inference_weight() == 12

    def test_it_honours_the_configured_capacity_per_cell(self):
        """The flag must reach the computation, not just be parsed."""
        server = _server(
            cells={
                "a": _cell(num_gpus_per_engine=4, is_serving=True),
                "b": _cell(num_gpus_per_engine=8, is_serving=True),
            },
            weight_per_cell=3,
        )
        assert server.inference_weight() == 6

    def test_a_fully_degraded_server_weighs_nothing(self):
        """Losing every cell must report zero rather than the last healthy value."""
        server = _server(
            cells={"a": _cell(num_gpus_per_engine=4, is_serving=False)},
            weight_per_cell=None,
        )
        assert server.inference_weight() == 0


class TestGetInferenceWeights:
    def test_it_reports_one_weight_per_model(self):
        """Each model sits behind its own router, so its capacity is reported separately."""
        controller = InferenceController.__new__(InferenceController)
        controller.servers = {
            "actor": _server(
                cells={
                    "a": _cell(num_gpus_per_engine=4, is_serving=True),
                    "b": _cell(num_gpus_per_engine=4, is_serving=True),
                },
                weight_per_cell=None,
            ),
            "ref": _server(
                cells={"c": _cell(num_gpus_per_engine=2, is_serving=True)},
                weight_per_cell=None,
            ),
        }
        assert controller.get_inference_weights() == {"actor": 8, "ref": 2}
