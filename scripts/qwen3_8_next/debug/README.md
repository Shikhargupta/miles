# Qwen3.8-Next debug & parity assets

Archived out of the production script dir once the bring-up stabilized
(first clean 5-rollout e2e: 2026-08-26). Nothing here runs in production.

- `megatron_logprobs.py` — offline harness: per-token logprobs, fwd/bwd parity
  (`--backward`), random-packed bwd stress (`--bwd-stress N`), env-gated probes
  (PARITY_REPEAT_BISECT / CHUNK_DOUBLE / FLA_SUBPROBE / CHUNK_O_BK).
- `run_parity_*.sh` — topology/backends matrix for the harness (TP/EP/PP, fla vs
  flashqla via GDN_BACKEND, torch vs triton via {QSA,HC,PLE}_BACKEND).
- `run_bwd_*.sh` — backward parity / SP+recompute / stress loop runners.
- `sglang_logprobs.py`, `compare_logprobs.py`, `make_parity_tokens.py` — the
  sglang side of a parity run and the token-set generator.
- `dump_patch_{sglang,megatron}.yaml` — dumper source-patcher configs; this is
  how dump points are injected now (inline parity_dump calls were removed from
  the model code).
- `repro_*.py` — minimal repros from the train-step SIGSEGV investigation:
  `repro_rpg_work.py` (the real one: ReloadableProcessGroup async-work path,
  torch-2.13 PyWorkHolder), `repro_hdo_segfault.py` and `repro_tms_lazyload.py`
  and `repro_tms_nccl.py` (exonerated suspects, kept as upstream-report seeds).
