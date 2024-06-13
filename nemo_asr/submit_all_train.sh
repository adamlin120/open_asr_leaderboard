#!/bin/bash

# Array of model IDs
MODEL_IDs=("nvidia/parakeet-rnnt-1.1b" "nvidia/parakeet-rnnt-0.6b" "nvidia/canary-1b")

# Array of dataset and split combinations
# DATASETS=(
#   "ami train"
#   "earnings22 train"
#   "gigaspeech train"
#   "librispeech train"
#   "spgispeech train"
#   "tedlium train"
#   "voxpopuli train"
#   "common_voice train"
# )
DATASETS=(
  "ami train[90%:]"
  "ami train[80%:90%]"
  "ami train[70%:80%]"
  "ami train[60%:70%]"
  "ami train[50%:60%]"
  "ami train[40%:50%]"
  "ami train[30%:40%]"
  "ami train[20%:30%]"
  "ami train[10%:20%]"
  "ami train[:10%]"
  "ami test"
  "earnings22 train[90%:]"
  "earnings22 train[80%:90%]"
  "earnings22 train[70%:80%]"
  "earnings22 train[60%:70%]"
  "earnings22 train[50%:60%]"
  "earnings22 train[40%:50%]"
  "earnings22 train[30%:40%]"
  "earnings22 train[20%:30%]"
  "earnings22 train[10%:20%]"
  "earnings22 train[:10%]"
  "earnings22 test"
  "gigaspeech train[90%:]"
  "gigaspeech train[80%:90%]"
  "gigaspeech train[70%:80%]"
  "gigaspeech train[60%:70%]"
  "gigaspeech train[50%:60%]"
  "gigaspeech train[40%:50%]"
  "gigaspeech train[30%:40%]"
  "gigaspeech train[20%:30%]"
  "gigaspeech train[10%:20%]"
  "gigaspeech train[:10%]"
  "gigaspeech test"
  "librispeech train[90%:]"
  "librispeech train[80%:90%]"
  "librispeech train[70%:80%]"
  "librispeech train[60%:70%]"
  "librispeech train[50%:60%]"
  "librispeech train[40%:50%]"
  "librispeech train[30%:40%]"
  "librispeech train[20%:30%]"
  "librispeech train[10%:20%]"
  "librispeech train[:10%]"
  "librispeech test.clean"
  "librispeech test.other"
  "spgispeech train[90%:]"
  "spgispeech train[80%:90%]"
  "spgispeech train[70%:80%]"
  "spgispeech train[60%:70%]"
  "spgispeech train[50%:60%]"
  "spgispeech train[40%:50%]"
  "spgispeech train[30%:40%]"
  "spgispeech train[20%:30%]"
  "spgispeech train[10%:20%]"
  "spgispeech train[:10%]"
  "spgispeech test"
  "tedlium train[90%:]"
  "tedlium train[80%:90%]"
  "tedlium train[70%:80%]"
  "tedlium train[60%:70%]"
  "tedlium train[50%:60%]"
  "tedlium train[40%:50%]"
  "tedlium train[30%:40%]"
  "tedlium train[20%:30%]"
  "tedlium train[10%:20%]"
  "tedlium train[:10%]"
  "tedlium test"
  "voxpopuli train[90%:]"
  "voxpopuli train[80%:90%]"
  "voxpopuli train[70%:80%]"
  "voxpopuli train[60%:70%]"
  "voxpopuli train[50%:60%]"
  "voxpopuli train[40%:50%]"
  "voxpopuli train[30%:40%]"
  "voxpopuli train[20%:30%]"
  "voxpopuli train[10%:20%]"
  "voxpopuli train[:10%]"
  "voxpopuli test"
  "common_voice train[90%:]"
  "common_voice train[80%:90%]"
  "common_voice train[70%:80%]"
  "common_voice train[60%:70%]"
  "common_voice train[50%:60%]"
  "common_voice train[40%:50%]"
  "common_voice train[30%:40%]"
  "common_voice train[20%:30%]"
  "common_voice train[10%:20%]"
  "common_voice train[:10%]"
  "common_voice test"
)

# Set the dataset path
DATASET_PATH="open-asr-leaderboard/datasets"

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