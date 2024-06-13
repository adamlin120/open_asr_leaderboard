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

python run_neko_eval.py --asr_dataset_config_name ${1} --model_name ${2}