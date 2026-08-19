from __future__ import annotations

from kubernetes_asyncio import client

from miles.utils.external_utils.colocate_pairing.config import PairingConfig

_GPU_RESOURCE = "nvidia.com/gpu"


class NodeWidthChecker:
    @classmethod
    def from_config(cls, *, config: PairingConfig, core_v1: client.CoreV1Api) -> NodeWidthChecker:
        widths = {pool.layout.num_gpus_per_node for pool in config.inference_pools}
        assert len(widths) == 1, (
            f"the inference pools were rendered against different node widths {sorted(widths)}, so no single "
            f"number describes the nodes this run's trainers hold and none can be checked against them"
        )
        return cls(core_v1=core_v1, num_gpus_per_node=widths.pop())

    def __init__(self, *, core_v1: client.CoreV1Api, num_gpus_per_node: int) -> None:
        self._core_v1 = core_v1
        self._num_gpus_per_node = num_gpus_per_node
        self._nodes_of_the_configured_width: set[str] = set()

    async def assert_node_is_as_wide_as_configured(self, node_name: str) -> None:
        if node_name in self._nodes_of_the_configured_width:
            return

        node = await self._core_v1.read_node(name=node_name)
        allocatable = int((node.status.allocatable or {}).get(_GPU_RESOURCE, 0))
        assert allocatable == self._num_gpus_per_node, (
            f"node {node_name} allocates {allocatable} {_GPU_RESOURCE} but this run was configured for "
            f"{self._num_gpus_per_node} gpus per node, so a colocated engine's card is computed against a node "
            f"width the node does not have and the trainer pod beside it does not hold every card it is given; "
            f"device plugin time slicing and mig inflate this count, so a node reporting them cannot host colocate"
        )
        self._nodes_of_the_configured_width.add(node_name)
