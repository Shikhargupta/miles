"""Frozen, host-resident, TP-row-sharded PLE n-gram table.

51.2 B parameters / 102.4 GB bf16 -- 28.4% of the model -- and every token reads
exactly 16 rows of it, 0.000005% of the table. Three consequences follow, and the
third is the one that matters for changing TP or PP later.

**Frozen.** Dense training would want 102.4 GB of gradient plus 614 GB of Adam
state and master weights, and all-reducing a 102 GB gradient every step would
swamp the compute. Freezing is also free for train/inference consistency: a frozen
table is bitwise identical between training and rollout, so it contributes exactly
zero to the logprob gap. (Full-parameter training of it wants row-sharded sparse
gradients, DLRM-style, which is its own project.)

**Host-resident.** Matches sglang's ``ple_offload_embedding=True``: pinned CPU
memory read by a kernel through a raw host pointer, never staged to HBM. On GB300
the Grace-Blackwell NVLink-C2C link makes that far cheaper than the PCIe hop the
pattern was designed around.

**Not a checkpointed parameter.** Registered non-persistent and loaded straight
from the HF safetensors at init. This is what keeps resharding cheap: torch_dist
reshards by splitting each parameter's shards, and re-splitting a row-sharded 51 B
table on every TP change is the single hardest thing in this model to reshard --
for content that never changes. Reading from the HF layout instead makes TP size
pure arithmetic (which of the 128 fixed shards does my row range cover), keeps the
torch_dist checkpoint at 257.6 GB instead of 360 GB, and means train->rollout
weight sync never has to carry it.
"""

import json
import logging
import struct

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.kernel.ple_gather import gather_ple_rows
from miles_plugins.models.qwen3_8_next.ops.ple_hash import ngram_hash_ids


logger = logging.getLogger(__name__)

_WEIGHT_MAP_CACHE: dict[str, dict] = {}


def _weight_map(hf_checkpoint: str) -> dict:
    """The checkpoint's name->file map, parsed once per path per process."""
    if hf_checkpoint not in _WEIGHT_MAP_CACHE:
        with open(f"{hf_checkpoint}/model.safetensors.index.json") as f:
            _WEIGHT_MAP_CACHE[hf_checkpoint] = json.load(f)["weight_map"]
    return _WEIGHT_MAP_CACHE[hf_checkpoint]


def _safetensors_slice(path: str, name: str, header_cache: dict) -> tuple[int, int, list[int]]:
    """Byte range and shape of one tensor, from the safetensors header."""
    if path not in header_cache:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header_cache[path] = (json.loads(f.read(n)), 8 + n)
    header, base = header_cache[path]
    meta = header[name]
    start, end = meta["data_offsets"]
    return base + start, base + end, meta["shape"]


class Qwen38NextFrozenNGramEmbedding(MegatronModule):
    """Frozen host-resident n-gram table, row-sharded over tensor parallelism."""

    def __init__(self, config: TransformerConfig, layer_number: int, tp_group=None):
        super().__init__(config)
        # Megatron numbers transformer layers from 1; the checkpoint indexes them
        # from 0. Getting this wrong does not fail at the table (every shard has the
        # same shape) -- it only fails here, on the metadata lookup, which is one
        # reason the metadata is loaded eagerly.
        self.layer_number = layer_number
        self.hf_layer_index = layer_number - 1
        heads = self._num_heads(config)
        self.embedding_dim = config.qwen3_8_next_ple_embed_dim // heads
        self.num_shards = config.qwen3_8_next_split_ngram_parts
        self.ngram_size = config.qwen3_8_next_ngram_size
        self._heads_per_ngram = config.qwen3_8_next_heads_per_ngram
        self.eos_token_id = getattr(config, "qwen3_8_next_eos_token_id", 0)
        self.tp_group = tp_group

        tp_size = tp_group.size() if tp_group is not None else 1
        tp_rank = tp_group.rank() if tp_group is not None else 0
        if self.num_shards % tp_size:
            raise ValueError(
                f"split_ngram_parts={self.num_shards} must be divisible by the "
                f"tensor-parallel size {tp_size}: shards are assigned whole so that "
                "changing TP only changes which of the fixed HF shards a rank reads."
            )
        self.shards_per_rank = self.num_shards // tp_size
        self.shard_ids = list(
            range(tp_rank * self.shards_per_rank, (tp_rank + 1) * self.shards_per_rank)
        )

        # Shard height comes from the checkpoint, not the config: the 320,001,446
        # hashed rows are rounded up to 128 x 2,500,012, so the last shard carries
        # padding that no arithmetic on the config would predict.
        self.rows_per_shard = getattr(config, "qwen3_8_next_ngram_rows_per_shard", None)
        if self.rows_per_shard is None:
            hf = getattr(config, "qwen3_8_next_hf_checkpoint", None)
            if hf is not None:
                self.rows_per_shard = self._shard_height_from_hf(hf, layer_number - 1)
        if self.rows_per_shard is None:
            # Last resort, and known to be slightly wrong: each hash head's vocab is
            # a distinct prime just above ngram_vocab_size_base (20000003, 20000023,
            # ...), so the real total is 320,001,446 rather than 16 x 20,000,000, and
            # this rounds to 2,500,000 per shard instead of 2,500,012. That
            # 12-row-per-shard drift misaligns the whole table. Only reachable when
            # initialising from scratch with no checkpoint to read.
            total = heads * config.qwen3_8_next_ngram_vocab_size_base
            self.rows_per_shard = -(-total // self.num_shards)
        self.row_start = self.shard_ids[0] * self.rows_per_shard
        self.row_end = (self.shard_ids[-1] + 1) * self.rows_per_shard

        # A plain attribute, deliberately NOT register_buffer. A registered buffer
        # follows the module through .cuda(), and Megatron moves the model to the
        # device after construction -- which silently relocated this 25.6 GB/rank
        # table to HBM and defeated the entire host-resident design. It fit, so
        # nothing failed; the only symptom was a numpy() call refusing a cuda
        # tensor. Nothing is lost by not registering it: it is excluded from
        # state_dict anyway (that is the point -- it must never enter the torch_dist
        # checkpoint or need resharding) and its loading is managed here.
        self.table = torch.empty(
            (self.row_end - self.row_start, self.embedding_dim),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self._loaded = False
        # Path to load from, stashed on the config by the spec. Loading is lazy: the
        # weight converter builds the model but never runs a forward, and reading
        # 102 GB it is not going to save would be pure waste.
        self._hf_checkpoint = getattr(config, "qwen3_8_next_hf_checkpoint", None)

        # Integer hash metadata: 35 int64 values total. Non-persistent for the same
        # reason as the table -- it comes from the HF checkpoint, not from training --
        # but loaded eagerly rather than lazily, because compute_ngram_ids is called
        # from outside this module (the model wrapper hashes the input tokens) and so
        # can run before any forward of the embedding would have triggered the table
        # load. mbridge walks buffers as well as parameters, so leaving these
        # persistent also made it demand mappings for them.
        self.register_buffer(
            "layer_multipliers", torch.zeros(self.ngram_size, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes", torch.zeros(heads, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.zeros(heads, dtype=torch.long), persistent=False
        )

        # After the buffers exist, not before.
        if self._hf_checkpoint is not None:
            self.load_metadata_from_hf(self._hf_checkpoint)

    @staticmethod
    def _shard_height_from_hf(hf_checkpoint: str, hf_layer_index: int) -> int | None:
        """Read shard 0's row count out of the safetensors header.

        Preferred over deriving it: the config's ngram_vocab_size_base is a base, not
        the actual per-head vocab, so arithmetic on it is off by a few rows per shard
        and silently misaligns every subsequent shard.
        """
        name = (
            f"model.language_model.layers.{hf_layer_index}.ple.ple_embedding"
            ".ngram_embedding.shard_0.weight"
        )
        try:
            index = _weight_map(hf_checkpoint)
            _, _, shape = _safetensors_slice(f"{hf_checkpoint}/{index[name]}", name, {})
            return int(shape[0])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                f"Could not read the PLE shard height from {hf_checkpoint} ({exc}); "
                "falling back to config arithmetic, which is KNOWN to drift by "
                "12 rows/shard and misalign the table (see __init__)."
            )
            return None

    @staticmethod
    def _num_heads(config: TransformerConfig) -> int:
        return (config.qwen3_8_next_ngram_size - 1) * config.qwen3_8_next_heads_per_ngram

    def _hf_index_and_prefix(self, hf_checkpoint: str):
        prefix = f"model.language_model.layers.{self.hf_layer_index}.ple.ple_embedding"
        return _weight_map(hf_checkpoint), prefix

    def load_metadata_from_hf(self, hf_checkpoint: str) -> None:
        """Load the three integer tensors that parameterise the hash.

        Eager, because the ids are computed outside this module and the hash is
        meaningless with zeros: every token would hash to row `offset`.
        """
        index, prefix = self._hf_index_and_prefix(hf_checkpoint)
        cache: dict = {}
        for buf_name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
            name = f"{prefix}.{buf_name}"
            path = f"{hf_checkpoint}/{index[name]}"
            start, end, shape = _safetensors_slice(path, name, cache)
            with open(path, "rb") as f:
                f.seek(start)
                raw = f.read(end - start)
            vals = torch.frombuffer(bytearray(raw), dtype=torch.int64).clone()
            getattr(self, buf_name).copy_(vals.reshape(shape))

    def load_from_hf(self, hf_checkpoint: str) -> None:
        """Fill the table from the HF safetensors.

        Reads only this rank's shards. Changing TP changes which shards that is and
        nothing else -- no checkpoint resharding, because the HF layout's 128 shards
        are fixed and TP-agnostic.
        """
        index, prefix = self._hf_index_and_prefix(hf_checkpoint)
        cache: dict = {}

        for i, shard_id in enumerate(self.shard_ids):
            name = f"{prefix}.ngram_embedding.shard_{shard_id}.weight"
            path = f"{hf_checkpoint}/{index[name]}"
            start, end, shape = _safetensors_slice(path, name, cache)
            rows = i * self.rows_per_shard
            dst = self.table[rows : rows + shape[0]]
            assert tuple(dst.shape) == tuple(shape), f"{name}: {tuple(dst.shape)} vs {shape}"
            with open(path, "rb") as f:
                f.seek(start)
                # Read straight into the pinned buffer's memory.
                mv = memoryview(dst.view(torch.uint8).reshape(-1).numpy())  # type: ignore[arg-type]
                f.readinto(mv)

        self._loaded = True

    def compute_ngram_ids(self, contexts: Tensor) -> Tensor:
        """``[T, ngram_size]`` sliding windows -> ``[T, n_heads]`` row ids."""
        return ngram_hash_ids(
            contexts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            self.ngram_size,
            self._heads_per_ngram,
            self.eos_token_id,
        )

    def forward(self, ids: Tensor) -> Tensor:
        """``[T, n_heads]`` int64 -> ``[T, n_heads * embedding_dim]`` bf16."""
        if not self._loaded:
            if self._hf_checkpoint is None:
                raise RuntimeError(
                    "PLE table was never loaded and no source is known. It is "
                    "deliberately not a checkpointed parameter (see this module's "
                    "docstring), so it is read from the HF safetensors on first use; "
                    "the spec normally records the path as "
                    "config.qwen3_8_next_hf_checkpoint. Running with a zero table "
                    "would change the logits without failing."
                )
            self.load_from_hf(self._hf_checkpoint)
        # Assert rather than trust: a registered buffer would have been moved to the
        # device by Megatron's .cuda(), and the kernel reads through a raw host
        # pointer, so a relocated table is silent corruption rather than an error.
        assert self.table.device.type == "cpu", (
            f"PLE table must stay on the host, found {self.table.device}. Something "
            "moved it -- most likely by registering it as a buffer again."
        )
        assert self.table.is_pinned(), "PLE table lost its pinning"
        rows = gather_ple_rows(self.table, ids, self.row_start, self.row_end)
        out = rows.flatten(start_dim=-2)
        if self.tp_group is not None and self.tp_group.size() > 1:
            # Each rank zeroed the rows it does not own.
            torch.distributed.all_reduce(out, group=self.tp_group)
        return out
