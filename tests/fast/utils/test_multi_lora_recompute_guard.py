"""Launch-time recompute guards for multi-LoRA (``validate_multi_lora_args``).

A checkpointed region is replayed grad-enabled only when its input requires
grad. Multi-LoRA trains adapter-only (frozen base), so full-layer recompute —
which wraps every adapter inside a checkpoint whose input never requires grad —
silently zeroes every adapter gradient (4xH200 GPT-OSS 20B evidence,
2026-08-12: grad_norm=0.0 on every step, zero trainer logprob delta). The same
mechanism applies to selective 'moe' recompute when the expert adapters are the
only trainable modules. These tests pin the launch-time refusals and the
supported selective configurations.
"""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.utils.multi_lora import validate_multi_lora_args


def _args(**overrides) -> SimpleNamespace:
    """Args rich enough to pass validate_multi_lora_args, mirroring
    test_tinker_predicates._full_args."""
    base = dict(
        tinker_backend=True,
        multi_lora_n_adapters=2,
        lora_rank=8,
        target_modules=["linear_qkv"],
        train_backend="megatron",
        pipeline_model_parallel_size=1,
        qkv_format="thd",
        experts_shared_outer_loras=False,
        optimizer="adam",
        colocate=False,
        indep_dp=False,
        ft_components=[],
        offload_train=False,
        enable_witness=False,
        sglang_tokenizer_worker_num=1,
        calculate_per_token_loss=False,
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        megatron_to_hf_mode="bridge",
        rollout_global_dataset=False,
        recompute_granularity=None,
        recompute_modules=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


EXPERT_TARGETS = ["gate_proj", "up_proj", "down_proj"]


class TestFullRecomputeRefused:
    def test_full_recompute_is_refused_for_any_targets(self):
        with pytest.raises(AssertionError, match="recompute-granularity full"):
            validate_multi_lora_args(_args(recompute_granularity="full"))

    def test_full_recompute_refusal_suggests_selective(self):
        with pytest.raises(AssertionError, match="selective"):
            validate_multi_lora_args(_args(recompute_granularity="full", target_modules=EXPERT_TARGETS))

    def test_refusal_happens_at_launch_not_after_gpu_time(self):
        # The guard must live in validate_multi_lora_args (driver launch), not in
        # the trainer: a refused config should never reach model build.
        args = _args(recompute_granularity="full")
        with pytest.raises(AssertionError):
            validate_multi_lora_args(args)


class TestSelectiveMoeModuleRefused:
    def test_moe_module_with_expert_targets_is_refused(self):
        with pytest.raises(AssertionError, match="moe_act"):
            validate_multi_lora_args(
                _args(
                    recompute_granularity="selective",
                    recompute_modules=["core_attn", "moe"],
                    target_modules=EXPERT_TARGETS,
                )
            )

    def test_moe_module_without_expert_targets_is_allowed(self):
        # Attention-only adapters sit outside the checkpointed MoE region; 'moe'
        # recompute is then a legitimate memory saver.
        validate_multi_lora_args(
            _args(
                recompute_granularity="selective",
                recompute_modules=["core_attn", "moe"],
                target_modules=["linear_qkv"],
            )
        )


class TestSupportedRecomputeConfigs:
    def test_no_recompute_is_allowed(self):
        validate_multi_lora_args(_args(target_modules=EXPERT_TARGETS))

    def test_selective_default_modules_is_allowed(self):
        # recompute_modules=None defaults to ['core_attn'] downstream.
        validate_multi_lora_args(_args(recompute_granularity="selective", target_modules=EXPERT_TARGETS))

    def test_selective_core_attn_moe_act_is_allowed_for_expert_targets(self):
        validate_multi_lora_args(
            _args(
                recompute_granularity="selective",
                recompute_modules=["core_attn", "moe_act"],
                target_modules=EXPERT_TARGETS,
            )
        )

    def test_absent_recompute_attrs_do_not_break_validation(self):
        args = _args()
        del args.recompute_granularity
        del args.recompute_modules
        validate_multi_lora_args(args)
