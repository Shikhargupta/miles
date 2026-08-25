"""Which checkpoint tensors can the bridge account for, and which cannot yet.

Expands every mapping the bridge declares across all layers and n-gram shards,
then diffs the result against the safetensors index. Answers "what is still
missing" mechanically instead of by inspection -- a weight silently left unmapped
is a model that loads with a parameter still at its init value.
"""
import json
import re
import sys

MODEL = "/data/models/Qwen3.8-Flash-Next"

from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge as B

cfg = json.load(open(f"{MODEL}/config.json"))
text = cfg["text_config"]
num_layers = text["num_hidden_layers"]
num_shards = text["split_ngram_parts"]
ple_layers = sorted({int(i) - 1 for i in text.get("ple_layer_ids") or []})
layer_types = text["layer_types"]
mtp_layers = text.get("mtp_num_hidden_layers", 0)
num_experts = text.get("num_experts", 0)

produced = set()

def add(template, **kw):
    produced.add(template.format(**kw))

for name, targets in list(B._DIRECT_MAPPING.items()):
    produced.add(targets if isinstance(targets, str) else targets[0])

per_layer = {}
per_layer.update(B._ATTENTION_MAPPING)
per_layer.update(B._MLP_MAPPING)
per_layer.update(B._OTHER_MAPPING)

# Mappings the model never asks for on this checkpoint. mbridge resolves names
# only for parameters the model actually creates, so declaring these is harmless --
# they are inherited from Qwen3_5Bridge, which also covers dense and biased
# variants. Excluded here so "mapped but not in ckpt" stays a real signal.
DENSE_MLP = ("mlp.linear_fc1.weight", "mlp.linear_fc2.weight")
BIASED = ("self_attention.linear_qkv.bias",)

for layer in range(num_layers):
    is_linear = layer_types[layer] == "linear_attention"
    for key, targets in per_layer.items():
        # PLE only exists on its own layers
        if key.startswith("ple.") and layer not in ple_layers:
            continue
        # the indexer and the dense q/k/v/o ride on full-attention layers only
        if ("indexer" in key or key.startswith("self_attention.self_attn.")) and is_linear:
            continue
        if key.startswith("self_attention.linear_attn.") and not is_linear:
            continue
        # On a linear-attention layer the whole standard attention block is
        # replaced by the gated-delta-net module, so Megatron's fused names
        # (linear_qkv / linear_proj) and the QK norms are never requested there.
        if is_linear and key in (
            "self_attention.linear_qkv.weight",
            "self_attention.linear_proj.weight",
            "self_attention.q_layernorm.weight",
            "self_attention.k_layernorm.weight",
        ):
            continue
        # a MoE checkpoint has no dense MLP, and attention_bias is False here
        if num_experts and key in DENSE_MLP:
            continue
        if key in BIASED:
            continue
        for t in (targets if isinstance(targets, list) else [targets]):
            if "{expert_id}" in t:
                for e in range(num_experts):
                    add(t, layer_number=layer, expert_id=e)
            else:
                add(t, layer_number=layer)

vision_depth = cfg.get("vision_config", {}).get("depth", 0)
for target in B._VISION_DIRECT_MAPPING.values():
    produced.add(target)
for layer in range(vision_depth):
    for targets in B._VISION_LAYER_MAPPING.values():
        for t in targets:
            produced.add(t.format(layer_number=layer))

for layer in ple_layers:
    for shard in range(num_shards):
        produced.add(
            f"model.language_model.layers.{layer}.ple.ple_embedding."
            f"ngram_embedding.shard_{shard}.weight"
        )

ckpt = set(json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"])

missing = sorted(ckpt - produced)
extra = sorted(produced - ckpt)

print(f"checkpoint tensors      : {len(ckpt)}")
print(f"claimed by the bridge   : {len(ckpt & produced)}  ({100*len(ckpt&produced)/len(ckpt):.1f}%)")
print(f"NOT claimed             : {len(missing)}  ({100*len(missing)/len(ckpt):.1f}%)")
print(f"mapped but not in ckpt  : {len(extra)}  (mapping targets nothing -- would fail at load)")

def group(keys):
    buckets = {}
    for k in keys:
        g = re.sub(r"\.\d+\.", ".{}.", k)
        g = re.sub(r"shard_\d+", "shard_{}", g)
        g = re.sub(r"\.\d+$", ".{}", g)
        buckets.setdefault(g, 0)
        buckets[g] += 1
    return sorted(buckets.items(), key=lambda kv: -kv[1])

if missing:
    print("\n=== NOT claimed, by pattern ===")
    for pat, n in group(missing):
        print(f"  {n:5d}  {pat}")
if extra:
    print("\n=== mapped but absent from the checkpoint, by pattern ===")
    for pat, n in group(extra)[:20]:
        print(f"  {n:5d}  {pat}")

print()
print("VERDICT=" + ("COMPLETE" if not missing and not extra else "INCOMPLETE"))
