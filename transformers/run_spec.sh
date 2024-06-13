#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

# MODEL_IDs=("openai/whisper-tiny.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-medium.en" "openai/whisper-large" "openai/whisper-large-v2")
MODEL_IDs=("openai/whisper-large-v2")
BATCH_SIZE=64

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="ami" \
        --split="test" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="earnings22" \
        --split="test" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="spgispeech" \
        --split="test" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="tedlium" \
        --split="test" \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
         --no-streaming \
        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=2 \
        --no-streaming \
        --batch_size=${BATCH_SIZE} \

        --max_eval_samples=-1

    python run_speculative_correction.py \
        --model_id=${MODEL_ID} \
        --dataset_path="open-asr-leaderboard/datasets-test-only" \
        --dataset="common_voice" \
        --split="test" \
        --no-streaming \
        --device=2 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
