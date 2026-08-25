"""Per-token logprobs from the Megatron model, for parity against sglang.

This is the measurement the whole exercise is aimed at: a mapping error, a missing
hyper-connection, a layernorm left at its init value, or an off-by-one in the PLE
hash all show up here as logprobs that do not match the rollout engine. A
tensor-by-tensor round trip through the bridge cannot catch any of them -- it
writes and reads with the same mapping, so it is close to a tautology.

Run under torchrun with the same parallelism the checkpoint was saved with.
Writes {out}/megatron_logprobs.pt with the token ids and per-token logprobs so the
comparison itself stays out of this process.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True, help="torch_dist checkpoint dir")
    p.add_argument("--hf-checkpoint", required=True)
    p.add_argument("--tokens", required=True, help=".pt file with a 1-D int64 tensor of input ids")
    p.add_argument("--out", required=True)
    return p.parse_known_args()[0]


def main():
    args = parse()

    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.training import get_model
    from megatron.core.enums import ModelType

    import miles_plugins.mbridge  # noqa: F401
    from miles.backends.megatron_utils.arguments import set_default_megatron_args
    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    margs = set_default_megatron_args(parse_args())
    margs.load = args.load
    margs.hf_checkpoint = args.hf_checkpoint
    margs.micro_batch_size = 1
    margs.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))
    validate_args(margs)
    init(margs)

    provider = get_model_provider_func(margs)
    model = get_model(provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    for m in model:
        m.eval()

    ids = torch.load(args.tokens).to(torch.long).cuda()
    seq = ids.numel()
    # Megatron wants [b, s]; the model is causal so one sequence is enough.
    input_ids = ids.view(1, seq)
    position_ids = torch.arange(seq, device=ids.device).view(1, seq)
    attention_mask = None

    with torch.no_grad():
        out = model[0](input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    logits = out if isinstance(out, torch.Tensor) else out[0]
    # [b, s, v] -> logprob of each *next* token, so the comparison lines up with
    # what an inference engine reports for the same prompt.
    logprobs = F.log_softmax(logits.float(), dim=-1)
    tgt = input_ids[:, 1:]
    picked = logprobs[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    if dist.get_rank() == 0:
        os.makedirs(args.out, exist_ok=True)
        torch.save(
            {"input_ids": ids.cpu(), "logprobs": picked[0].cpu(), "logits_shape": tuple(logits.shape)},
            os.path.join(args.out, "megatron_logprobs.pt"),
        )
        print(f"MEGATRON_LOGPROBS_OK seq={seq} logits={tuple(logits.shape)}")
    dist.barrier()


if __name__ == "__main__":
    main()
