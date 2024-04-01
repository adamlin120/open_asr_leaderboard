#!/bin/bash
#SBATCH --job-name=asr-eval-$1-$2
#SBATCH --output=logs/asr-eval-$1-$2-%j.out
#SBATCH --error=logs/asr-eval-$1-$2-%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("nvidia/canary-1b")
BATCH_SIZE=64
MAX_SAMPLES=-1
num_models=${#MODEL_IDs[@]}

dataset=$1
split=$2

for (( i=0; i<${num_models}; i++ )); do
    MODEL_ID=${MODEL_IDs[$i]}
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets" \
        --dataset=${dataset} \
        --split=${split} \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=${MAX_SAMPLES} \
        --no-streaming
done