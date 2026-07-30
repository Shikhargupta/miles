import random

from miles.utils.http_utils import find_available_port


def resolve_session_server_ports(raw: list[int] | None) -> list[int]:
    """Resolve the ``--session-server-port`` value into the ports to serve on.

    None: one auto-allocated port. One value: a single server on that port.
    Two values: the half-open range [start, end), one server per port.
    """
    if raw is None:
        return [find_available_port(random.randint(5000, 6000))]
    if len(raw) == 1:
        return raw
    if len(raw) == 2:
        start, end = raw
        if start >= end:
            raise ValueError(f"--session-server-port range [{start}, {end}) is empty.")
        return list(range(start, end))
    raise ValueError(f"--session-server-port takes one port or a start/end range, got {len(raw)} values: {raw}")


# TODO: temporary
def compute_num_session_server_ports(args) -> int:
    return len(resolve_session_server_ports(getattr(args, "session_server_port", None)))
