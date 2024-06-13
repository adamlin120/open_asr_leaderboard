#!/bin/bash

# Array of model IDs
MODEL_IDs=("nvidia/parakeet-ctc-1.1b" "nvidia/parakeet-ctc-0.6b" "nvidia/canary-1b")

# Array of dataset and split combinations
DATASETS=(
  "ami test"
  "earnings22 test"
  "gigaspeech test"
  "librispeech test.clean"
  "librispeech test.other"
  "spgispeech test"
  "tedlium test"
  "voxpopuli test"
  "common_voice test"
)

# Set the dataset path
DATASET_PATH="open-asr-leaderboard/datasets-test-only"

# Loop through each model
for MODEL_ID in "${MODEL_IDs[@]}"
do
  # Loop through each dataset and split combination
  for DATASET_SPLIT in "${DATASETS[@]}"
  do
    # Extract the dataset and split
    DATASET=$(echo $DATASET_SPLIT | cut -d' ' -f1)
    SPLIT=$(echo $DATASET_SPLIT | cut -d' ' -f2)
    
    # Submit the job using sbatch
    sbatch asr_eval.slurm $DATASET $SPLIT $DATASET_PATH $MODEL_ID
  done
done