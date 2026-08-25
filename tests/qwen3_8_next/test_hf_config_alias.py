"""Can transformers load Qwen3.8-Next's config once the aliases are registered?

The checkpoint declares model_type qwen4_exp with a nested qwen4_exp_text, neither
of which transformers 5.12.1 knows, and it carries no remote code -- so AutoConfig
raises before anything else gets a chance to run. Both levels need an alias: a
composite config resolves its sub-config class from the nested model_type, so
registering only the outer one still fails on text_config.
"""
import sys

from miles.utils.hf_config import load_hf_config, register_hf_config_aliases

MODEL = "/data/models/Qwen3.8-Flash-Next"

register_hf_config_aliases()
cfg = load_hf_config(MODEL)
print(f"outer model_type : {cfg.model_type}")
tc = cfg.text_config
print(f"text  model_type : {getattr(tc, 'model_type', None)}")

ok = True
checks = {
    "num_hidden_layers": 48,
    "hidden_size": 2560,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "vocab_size": 248320,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    # the fields the base class has never heard of -- these are the ones that
    # would silently vanish if transformers dropped unknown kwargs
    "hc_count": 4,
    "hc_lowrank": 320,
    "ple_embed_dim": 2560,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "split_ngram_parts": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_n_heads": 4,
    "indexer_head_dim": 128,
}
for k, want in checks.items():
    got = getattr(tc, k, None)
    good = got == want
    ok &= good
    print(f"  {k:26s} {str(got):>10s}  expect {str(want):>10s}  {'ok' if good else 'MISMATCH'}")

print(f"  ple_layer_ids              {getattr(tc, 'ple_layer_ids', None)}")
print(f"  vision depth               {getattr(cfg.vision_config, 'depth', None)}")
print()
print("VERDICT=" + ("PASS" if ok else "FAIL"))
