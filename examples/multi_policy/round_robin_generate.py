from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.megatron_config import resolve_megatron_config


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    model_ids = resolve_megatron_config(input.args).model_ids
    input.sample.trainer_model_id = model_ids[(input.sample.group_index or 0) % len(model_ids)]
    return await single_turn_generate(input)
