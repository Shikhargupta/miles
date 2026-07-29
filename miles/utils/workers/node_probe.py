from miles.utils.misc import get_current_node_ip, get_free_port


class NodeProbeMixin:
    @staticmethod
    def _get_node_ip() -> str:
        return get_current_node_ip()

    @staticmethod
    def _get_free_port_block(*, start_port: int, count: int) -> int:
        return get_free_port(start_port=start_port, consecutive=count)
