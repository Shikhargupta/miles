def compute_cell_id(*, spec_name: str, cell_index: int) -> str:
    return f"{spec_name}-{cell_index}"


def compute_worker_name(*, spec_name: str, cell_index: int, worker_index: int) -> str:
    return f"{compute_cell_id(spec_name=spec_name, cell_index=cell_index)}-{worker_index}"
