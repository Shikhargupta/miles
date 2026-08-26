"""Side channel carrying the current forward's n-gram row ids to the PLE layer.

PLE needs token ids to hash, but a Megatron transformer layer only sees
``hidden_states`` -- by the time the state reaches layer 1 the token ids are three
frames up, and under pipeline parallelism they may not be on this rank at all. The
inference side has the same problem and solves it the same way (sglang threads a
``_PLEBatch`` through ``get_req_to_token_pool()``), so this mirrors it rather than
changing Megatron's layer signatures, which would mean touching
``TransformerBlock.forward`` and ``HyperConnectionTransformerLayer.forward`` and
then threading the ids across pipeline stages.

The tradeoff is implicit state, and implicit state that silently defaults is how
you ship a model that runs and is quietly wrong. So there is no default: a PLE
forward with nothing set raises. Construction-only paths (weight conversion,
shape audits) never call forward, so they are unaffected; ``allow_missing`` exists
for tests that deliberately want the no-PLE path.
"""

import contextlib
import threading

import torch

_state = threading.local()


class PLEContextError(RuntimeError):
    pass


@contextlib.contextmanager
def ple_forward_context(ngram_ids: torch.Tensor, cu_seqlens: torch.Tensor | None = None):
    """Publish this forward's ``[T, n_heads]`` n-gram row ids for the PLE layer.

    ``cu_seqlens`` lets the PLE short conv respect document boundaries; pass it for
    packed (THD) batches, which is how miles feeds training data.
    """
    prev = getattr(_state, "batch", None)
    _state.batch = (ngram_ids, cu_seqlens)
    try:
        yield
    finally:
        _state.batch = prev


def current_ple_batch(*, allow_missing: bool = False):
    batch = getattr(_state, "batch", None)
    if batch is None and not allow_missing:
        raise PLEContextError(
            "PLE ran with no n-gram ids published. Wrap the model forward in "
            "ple_forward_context(ngram_ids, cu_seqlens); the ids come from "
            "Qwen38NextFrozenNGramEmbedding.compute_ngram_ids on the input tokens. "
            "Silently skipping PLE would leave layer 1 missing its contextual "
            "increment, which changes the logits without failing."
        )
    return batch


def publish_ple_batch(ngram_ids: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> None:
    """Hook-style publication for callers that cannot hold a context manager open.

    The model-provider forward hooks use this: a ``register_forward_pre_hook`` has
    no scope that survives until the matching post-hook, so the pair
    publish/clear replaces the ``with`` block. Same thread-local, same contract.
    """
    _state.batch = (ngram_ids, cu_seqlens)


def clear_ple_batch() -> None:
    _state.batch = None
