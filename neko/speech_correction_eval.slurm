#!/bin/bash
#SBATCH --job-name=speech_correction_eval
#SBATCH --output=speech_correction_eval.out
#SBATCH --error=speech_correction_eval.err
#SBATCH --partition=defq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Load necessary modules
ml slurm

# Activate your virtual environment (if applicable)
source ~/.bashrc
conda init bash
conda activate asr

# Set the PYTHONPATH environment variable
export PYTHONPATH="..":$PYTHONPATH

# Run the Python script with desired arguments
python run_neko_eval.py \
    --results_file $1 \
    --dataset_name yentinglin/asr \
    --model_name $2