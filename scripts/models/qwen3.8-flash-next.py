from model_args_utils import moe_layer_freq


NLAYERS = 48
FIRST_K_DENSE_REPLACE = 0


def model_args() -> str:
    """Qwen3.8-Flash-Next: 180 B total, ~7.4 B active per token.

    Shapes are read straight off the released config.json. Two things are NOT here
    because Megatron has no CLI flag for them and the spec derives them from the
    checkpoint instead: the hyper-connection fields (hc_count / hc_lowrank) and the
    PLE and QSA fields. See _apply_qwen3_8_next_config in
    miles_plugins/models/qwen3_8_next/qwen3_8_next.py.

    --mtp-num-layers is deliberately omitted for now: MTP's 31 tensors are not yet
    mapped, and its structure has an open question (mtp.fc_hidden is [2560, 2560], so
    the hidden fed to it is 2560 wide rather than the widened 10240, yet MTP also
    carries its own hyper_connection_mixer).
    """
    return (
        # package.module, not just the package: --spec imports the module and looks the
        # function up in it, and the package __init__ does not re-export it.
        "--spec miles_plugins.models.qwen3_8_next.qwen3_8_next get_qwen3_8_next_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 24 "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        "--num-layers 48 "
        "--hidden-size 2560 "
        "--ffn-hidden-size 640 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        # partial_rotary_factor = 0.25
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        # moe: 512 experts, 10 active
        "--moe-ffn-hidden-size 640 "
        "--moe-shared-expert-intermediate-size 640 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 10 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=NLAYERS, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 512 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
        # qwen3.5 lineage
        "--attention-output-gate "
        "--moe-shared-expert-gate "
    )
