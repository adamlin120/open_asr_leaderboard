#!/bin/bash

# Array of model IDs
# MODEL_IDs=("openai/whisper-medium" "openai/whisper-small" "openai/whisper-base" "openai/whisper-tiny" "distil-whisper/distil-large-v3" "distil-whisper/distil-medium.en" "distil-whisper/distil-small.en" "openai/whisper-large-v3" "openai/whisper-medium.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-tiny.en")
MODEL_IDs=("openai/whisper-large-v3" "openai/whisper-large-v2")

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
DATASET_PATH="yentinglin/asr_correction_datasets"

MAX_SAMPLES=-1

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
    sbatch asr_eval.slurm $DATASET $SPLIT $DATASET_PATH $MODEL_ID $MAX_SAMPLES
  done
done