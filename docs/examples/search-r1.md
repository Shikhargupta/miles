---
title: Search-R1 (Tool Use)
description: Train a model to issue search queries, integrate observations, and answer multi-turn QA.
---
**What you'll learn:** how to wire up a tool (web search) into a Miles training loop —
custom multi-turn rollout, observation interleaving, reward function, and TIS to keep
training stable when train ≠ inference.

This is a Miles-friendly reproduction of the original
[Search-R1](https://github.com/PeterGriffinJin/Search-R1).

## Prerequisites

* `radixark/miles:latest` container.
* Either a serper.dev API key (Google search backend) or ~135 GB free disk for the
  local Wikipedia retriever (see [appendix](#appendix-local-wikipedia-retriever)).
* You completed [Customization](/user-guide/customization) — this example uses a
  custom rollout function and reward.

## Files

```text
examples/search-r1/
├── generate_with_search.py       # custom rollout (multi-turn loop)
├── google_search_server.py       # serper.dev wrapper
├── local_search_server.py        # FastAPI server in front of FAISS index
├── local_dense_retriever/        # E5-base index/corpus downloader
├── qa_em_format.py               # exact-match reward
└── run_qwen2.5_3B.sh             # launch script
```

## Quick start

### 1. Set up environment

```bash
cd /root && git clone https://github.com/radixark/miles.git
cd miles && pip install -e . --no-deps && pip install chardet
```

### 2. Prepare data

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1 && pip install -e . --no-deps && pip install tensordict

WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources nq,hotpotqa
```

### 3. Convert the model

```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B
cd /root/miles
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-3B \
   --save           /root/Qwen2.5-3B_torch_dist
```

### 4. Run

```bash
bash examples/search-r1/run_qwen2.5_3B.sh
```

## Configuration

Open `generate_with_search.py` and edit `SEARCH_R1_CONFIGS`:

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",     # or "google"

    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },

    "google": {
        "api_key": "your_serper_key",
        "snippet_only": True,
        "proxy": None,
    },

    "return_logprob": True,        # required for TIS
    "format_score": 0.2,
}
```

## Walkthrough — multi-turn rollout

The custom rollout lives in `generate_with_search.py:generate`. The model emits
either `<search>query</search>` or `<answer>...</answer>`; search results come back
wrapped in `<information>...</information>`. Condensed:

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    response, response_token_ids, loss_mask = "", [], []

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        # 1. Model generates an action
        output = await post(url, {"text": sample.prompt + response,
                                  "sampling_params": sampling_params})
        cur_response_token_ids = ...   # token ids (and log probs) from the output
        response += output["text"]
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)  # model tokens count toward loss

        # 2. Parse the action and run the tool; <answer> ends the loop
        next_obs, done = await execute_predictions(output["text"])
        if done:
            break

        # 3. Feed the observation back, masked out of the loss
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)          # observation tokens MASKED OUT

    sample.response  = response
    sample.tokens    = prompt_tokens_ids + response_token_ids
    sample.loss_mask = loss_mask
    return sample
```

### The two crucial details

1. **Loss masking.** Tool/observation tokens get `loss_mask=0`. Without this, the model
   learns to *predict the search results*, which is both wrong and wildly unhelpful.
2. **Tokenization alignment.** The model must see and the trainer must score the
   *exact same tokens*. Pre-tokenizing vs. re-tokenizing at training time can drift —
   that's where the [chat template verifier](/user-guide/agentic-chat-template)
   matters.

## Walkthrough — reward

```python
async def reward_func(args, sample, **kwargs):
    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )
    return score
```

`compute_score_em` (in `qa_em_format.py`) extracts the `<answer>...</answer>` span and
exact-matches it against the label. `format_score=0.2` gives partial credit for the
correct shape even if the content is wrong — keeps gradient flowing during early
exploration.

## Enabling TIS

The trajectory mixes model tokens (we want gradients) with tool tokens (we don't).
Without correction, the implicit policy ratio in the GRPO objective is *off-policy* —
the search results came from a stochastic environment, not the model.

**Truncated Importance Sampling (TIS)** corrects for this. To enable:

1. Set `"return_logprob": True` in `SEARCH_R1_CONFIGS`.
2. Uncomment the TIS flags in `run_qwen2.5_3B.sh`:

```bash
GRPO_ARGS+=( --use-tis )
CUSTOM_ARGS+=(
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

When `return_logprob=True`, response post-processing is automatically disabled to keep
token / logp alignment.

## What to watch

```text
rollout/raw_reward                          trending up (EM + format credit)
perf/rollout_time                           grows with max_turns (each turn adds a retrieval RTT)
train/loss, train/pg_loss, train/grad_norm  the standard step-line metrics
train/train_rollout_logprob_abs_diff        train-vs-rollout drift (needs return_logprob)
```

If `train/train_rollout_logprob_abs_diff` keeps climbing, training and inference have
drifted too far apart. Lower `--lr` or shorten `max_turns`.

## Tuning knobs

| Knob | Effect |
|---|---|
| `max_turns` | More turns = more retrieval, more drift |
| `topk` | More retrieved snippets = longer context |
| `search_concurrency` | Cap on simultaneous tool calls (mind your QPS limit) |
| `format_score` | Partial credit for correct shape — higher = faster early shaping |

## Troubleshooting

| Problem | Fix |
|---|---|
| "Ray process stuck" | `rm -rf /root/.cache`, then `rm -rf /root/.*` if still stuck |
| Retriever 502 errors | `lsof -i :8000` — make sure your local server is alive |
| Conda activation collisions | Deactivate the `retriever` env before launching training |
| EM stays at 0 | Check the answer extractor — most often a regex mismatch |
| Loss masks shifted by one token | Tokenizer added a leading space; align with `add_special_tokens=False` |

## Variations

* **Use Google instead of local.** Set `"search_backend": "google"` and add an API key.
* **Different tool.** Replace `search_backend` with anything else — calculator, code
  exec, internal API. The pattern is identical.
* **Group RM.** With multiple trajectories per prompt (GRPO), enable `--group-rm` so
  rewards are computed in a batch.
* **Longer chains.** Bump `max_turns` to 8+ for deep-reasoning tasks. Watch how much
  of each trajectory is masked observation tokens — if observations dominate, the
  model is barely training.

## Appendix — local Wikipedia retriever

Heavy but completely offline. ~135 GB total disk and a separate conda env to avoid
conflicting with Miles.

### One-time setup

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n retriever python=3.10 -y && conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
              pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# 2. Index + corpus (~135 GB)
save_path=/root/Index
python /root/miles/examples/search-r1/local_dense_retriever/download.py \
    --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### Run the server

```bash
conda activate retriever
python /root/miles/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path /root/Index/e5_Flat.index \
    --corpus_path /root/Index/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

5–7 GB of GPU memory per GPU. First startup is slow (model + index load); subsequent
restarts are 1–2 minutes.

### Then launch training

```bash
conda deactivate              # don't train inside the retriever env!
cd /root/miles
bash examples/search-r1/run_qwen2.5_3B.sh
```
