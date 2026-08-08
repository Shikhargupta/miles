"""Sampling fields that Miles may pin for every request in a session."""

SESSION_REQUEST_OVERRIDE_KEYS = frozenset(
    {
        "custom_logit_processor",
        "custom_params",
        "ebnf",
        "frequency_penalty",
        "ignore_eos",
        "logit_bias",
        "max_completion_tokens",
        "max_tokens",
        "min_p",
        "min_tokens",
        "n",
        "presence_penalty",
        "regex",
        "repetition_penalty",
        "response_format",
        "seed",
        "stop",
        "stop_regex",
        "stop_token_ids",
        "temperature",
        "top_k",
        "top_p",
    }
)
