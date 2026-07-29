from miles.utils.workers.node_probe import NodeProbeMixin


class TestNodeProbeMixin:
    def test_get_node_ip_returns_nonempty_string(self):
        """The node ip probe answers with a usable address string."""
        node_ip = NodeProbeMixin._get_node_ip()
        assert isinstance(node_ip, str) and node_ip

    def test_get_free_port_block_returns_first_port_at_or_above_start(self):
        """A block request returns the first port of a free consecutive range."""
        first_port = NodeProbeMixin._get_free_port_block(start_port=15000, count=2)
        assert first_port >= 15000
