#!/usr/bin/env fish

set -l run_id 260820-25fff782
set -l run_name 260820-25fff782-glm47-t2-summary23
set -l script_dir (path resolve (dirname (status filename)))

python $script_dir/run.py \
    --skip-prepare \
    --run-id $run_id \
    --hf-checkpoint /cluster-storage/models/GLM-4.7-Flash \
    --ref-load /cluster-storage/models/GLM-4.7-Flash_torch_dist \
    --save-dir /scratch/$run_id/checkpoints \
    --save-interval 100 \
    --prompt-data $script_dir/tb2_23_tasks.jsonl \
    --max-seq-len 32768 \
    --num-rollout 100 \
    --rollout-batch-size 4 \
    --n-samples-per-prompt 8 \
    --global-batch-size 32 \
    --save-traces-dir /scratch/$run_id/traces \
    --agent-server-url http://agent-server:11000 \
    --agent-model-name model \
    --router-external-host devbox-gpu-260820-25fff782-glm47-t2-summary23-64a44c75 \
    --wandb-project glm47-flash-agentic \
    --wandb-team ch271828n-team \
    --wandb-run-name $run_name \
    --prometheus-run-name $run_name
