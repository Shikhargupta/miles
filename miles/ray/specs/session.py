from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec

_SESSION_SERVER_PORT = 30100


def compute_session_server_specs(args) -> list[CommandWorkerSpec]:
    if not args.use_session_server:
        return []

    return [
        CommandWorkerSpec(
            name="session-server",
            port_infos=[_session_server_port_info(args)],
            env_var=lambda: {},
            scheduling=SchedulingSpec(
                num_cells=_compute_num_session_servers(args),
                num_workers_per_cell=1,
                num_gpus_per_worker=0,
            ),
            launch_command="python -m miles.rollout.session.server --config-json {config_json}",
        )
    ]


def _compute_num_session_servers(args) -> int:
    raw = args.session_server_port
    if raw is None or len(raw) == 1:
        return 1
    if len(raw) == 2:
        start, end = raw
        assert start < end, f"--session-server-port range [{start}, {end}) is empty."
        return end - start
    raise ValueError(f"--session-server-port takes one port or a start/end range, got {len(raw)} values: {raw}")


def _session_server_port_info(args) -> PortInfo:
    raw = args.session_server_port
    return PortInfo(
        name="http",
        static_port=raw[0] if raw else _SESSION_SERVER_PORT,
        mode="per_worker",
        allow_dynamic=raw is None,
    )
