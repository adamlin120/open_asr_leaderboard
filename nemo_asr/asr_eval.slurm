#!/bin/bash
#SBATCH --job-name=asr-eval
#SBATCH --output=logs/asr-eval-%j.out
#SBATCH --error=logs/asr-eval-%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

export PYTHONPATH="..":$PYTHONPATH
source ~/.bashrc
conda init bash
source activate asr

BATCH_SIZE=64
MAX_SAMPLES=${5:--1}
dataset=$1
split=$2
dataset_path=$3
MODEL_ID=$4

python run_eval.py \
  --model_id=${MODEL_ID} \
  --dataset_path=${dataset_path} \
  --dataset=${dataset} \
  --split=${split} \
  --device=0 \
  --batch_size=${BATCH_SIZE} \
  --max_eval_samples=${MAX_SAMPLES} \
  --no-streaming
