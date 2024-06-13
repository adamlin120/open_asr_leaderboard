from datasets import Dataset
import os
from pathlib import Path
import pandas as pd

# export PYTHONPATH="..":$PYTHONPATH
os.environ["PYTHONPATH"] = "..:" + os.environ.get("PYTHONPATH", "")
from normalizer.data_utils import normalizer
from tqdm import tqdm

# add logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for file in tqdm(
    list(Path("results").glob("*.jsonl"))
    #  + list(Path("results").glob("MODEL_*_DATASET_open-asr-leaderboard-datasets-test-only_*_test.jsonl"))
):
    print(f"Processing {file}")

    name = file.stem
    # e.g. MODEL_nvidia-canary-1b_DATASET_open-asr-leaderboard-datasets_ami_train.jsonl
    # Parse model_name, dataset_name, split_name
    parts = name.split("_")
    model_name = parts[1]
    dataset_name = parts[4]
    split_name = parts[5]

    if split_name not in {"train", "dev", "test"}:
        logger.warning(
            f"Currently only support train, dev, test split, skipping {split_name} in {file}"
        )
        continue

    # Load JSONL data and convert to dict
    data = pd.read_json(file, lines=True)

    if "hypotheses" not in data.columns:
        # warning
        logger.warning(f"Missing hypotheses in {file}, skipping")
        continue

    # remove col: audio_filepath, duration
    data = data.drop(columns=["audio_filepath", "duration"])

    # remove col: pred_text
    if "pred_text" in data.columns:
        data = data.drop(columns=["pred_text"])

    # rename col: text -> output
    data = data.rename(columns={"text": "output"})

    # normalized hypotheses
    # e.g. data_utils.normalizer(sample["pred_text"])
    data["input"] = [repr([normalizer(x) for x in xs]) for xs in data["hypotheses"]]

    # add instruction
    data["instruction"] = (
        "The following text contains 5-best hypotheses from an Automatic Speech Recognition system. "
        "As part of a speech recognition task, please perform error correction on the hypotheses to "
        "generate the most accurate transcription of the spoken text."
    )

    # Create dataset
    dataset = Dataset.from_pandas(data)

    print(dataset)

    config_name = f"{model_name}_{dataset_name}"

    print(f"Uploading {config_name} {split_name} with {len(dataset)} samples")

    # Upload to HuggingFace
    dataset.push_to_hub("yentinglin/asr", config_name, split=split_name)
