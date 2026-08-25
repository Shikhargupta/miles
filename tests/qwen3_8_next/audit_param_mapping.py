"""Every parameter the model creates -- does the bridge know where it comes from?

The conversion loop finds these one at a time: build the model, start loading,
raise on the first unmapped name, fix, rebuild. This asks the same question about
all of them at once, on a structurally identical but tiny model, because parameter
*names* do not depend on layer count or expert count.

Structure that must be preserved for the names to be representative:
  * both layer types present (linear_attention and full_attention), so the GDN
    wrapper and the dense attention block both get built;
  * the PLE layer present;
  * MoE on, so the fused expert tensors appear rather than a dense MLP.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29551")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

torch.cuda.set_device(0)
dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state

parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

model_parallel_cuda_manual_seed(1234)

from megatron.core.transformer.spec_utils import ModuleSpec  # noqa: E402
from megatron.core.transformer.transformer_block import TransformerBlock  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles.utils.hf_config import load_hf_config, register_hf_config_aliases  # noqa: E402
from miles_plugins.models.qwen3_8_next.qwen3_8_next import (  # noqa: E402
    _apply_qwen3_8_next_config,
    _layer_types,
    get_qwen3_8_next_spec,
)

MODEL = "/data/models/Qwen3.8-Flash-Next"
NLAYERS = 8  # layers 0..7 -> 6 linear + 2 full (interval 4), PLE on layer 1


class _Args:
    hf_checkpoint = MODEL
    num_experts = 8


def build():
    register_hf_config_aliases()
    hf = load_hf_config(MODEL)
    text = hf.text_config

    config = TransformerConfig(
        num_layers=NLAYERS,
        hidden_size=text.hidden_size,
        num_attention_heads=text.num_attention_heads,
        num_query_groups=text.num_key_value_heads,
        kv_channels=text.head_dim,
        ffn_hidden_size=text.moe_intermediate_size,
        num_moe_experts=8,
        moe_ffn_hidden_size=text.moe_intermediate_size,
        moe_shared_expert_intermediate_size=text.shared_expert_intermediate_size,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_shared_expert_overlap=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        layernorm_epsilon=text.rms_norm_eps,
        sequence_parallel=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        pipeline_dtype=torch.bfloat16,
        attention_output_gate=True,
        qk_layernorm=True,
        # Must match scripts/models/qwen3.8-flash-next.py: the TransformerConfig
        # default is LayerNorm, whose bias has no counterpart in the checkpoint
        # (which carries only q_norm.weight / k_norm.weight), so leaving it out
        # produced two phantom unmapped parameters per full-attention layer.
        normalization="RMSNorm",
        # --apply-layernorm-1p in the model args maps to this: Megatron's norms then
        # use (1 + weight), which is the same zero-centered-gamma convention the
        # checkpoint stores its norm weights in.
        layernorm_zero_centered_gamma=True,
    )
    # the spec sets enable_hyper_connections and the PLE/QSA fields itself
    spec = get_qwen3_8_next_spec(_Args(), config, vp_stage=None)
    block = TransformerBlock(config=config, spec=spec, post_process=True, post_layer_norm=True)
    return config, block, text


def main():
    config, block, text = build()

    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    # Only the mapping tables are needed, so avoid AutoBridge's full construction.
    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    bridge.hf_config = load_hf_config(MODEL)

    names = [n for n, _ in block.named_parameters()]
    print(f"model built: {NLAYERS} layers, {len(names)} parameters")
    print(f"layer types: {_layer_types(text)[:NLAYERS]}")

    # Mapping coverage alone is not enough: mbridge only resolves names for
    # parameters the model actually creates, so a module that was written but never
    # wired into the spec makes its checkpoint tensors silently vanish rather than
    # raise. Check the structure is present, not just that it is mapped.
    import re

    patterns = sorted({re.sub(r"\.\d+\.", ".{}.", n) for n in names})
    print("=== distinct parameter patterns ===")
    for pat in patterns:
        print("   ", pat)

    expected_present = {
        # PLE hangs off the attention-site HC, so exclude it when counting HC params
        # or its tensors get attributed to the hyper-connection.
        "hyper-connection (attn)": r"self_attention_hyper_connection\.(?!ple\.)",
        "hyper-connection (mlp)": r"mlp_hyper_connection",
        "final contraction": r"hc_head_contraction",
        "PLE proj/norms/conv": r"\.ple\.(key_proj|value_proj|norm_|conv1d)",
        "QSA indexer": r"indexer",
        "GDN linear attention": r"linear_attn",
        "MoE experts": r"experts",
    }
    # The n-gram table is intentionally NOT a parameter: non-persistent buffer, read
    # from the HF safetensors on first use, so it never enters the checkpoint and
    # never needs resharding on a TP change. Assert that on purpose.
    table_params = [n for n in names if "ngram_embedding" in n]
    print(f"  n-gram table as parameter  {len(table_params):4d} params  "
          f"{'ok (excluded by design)' if not table_params else 'UNEXPECTED'}")
    print("=== structural presence ===")
    missing_structure = []
    for label, pat in expected_present.items():
        hits = [n for n in names if re.search(pat, n)]
        if not hits:
            missing_structure.append(label)
        print(f"  {label:26s} {len(hits):4d} params  {'ok' if hits else 'ABSENT'}")

    unmapped = []
    mapped = 0
    for n in names:
        full = f"decoder.{n}"
        try:
            bridge._weight_name_mapping_mcore_to_hf(full)
            mapped += 1
        except NotImplementedError:
            unmapped.append(full)
        except Exception as e:  # a mapping that exists but blows up is also a bug
            unmapped.append(f"{full}   [{type(e).__name__}: {e}]")

    print(f"mapped   : {mapped}")
    print(f"UNMAPPED : {len(unmapped)}")
    for u in unmapped:
        print("   ", u)
    print()
    if table_params:
        missing_structure.append("n-gram table leaked into parameters")
    if missing_structure:
        print(f"PROBLEMS: {missing_structure}")
    print("VERDICT=" + ("PASS" if not unmapped and not missing_structure else "FAIL"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("VERDICT=BUILD_FAILED")
