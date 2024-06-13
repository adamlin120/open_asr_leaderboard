import os
from pprint import pprint
from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from tqdm import tqdm
from nemo_asr.prompt_utils import generate_instructions, InputType, OutputType

dataset_to_pnc_type = {
    "ami": "Punctuated & Cased",
    "earnings22": "Punctuated & Cased",
    "spgispeech": "Punctuated & Cased",
    "librispeech": "Normalized",
    "gigaspeech": "Punctuated",
    "tedlium": "Normalized",
    "voxpopuli": "Punctuated",
    "commonvoice": "Punctuated & Cased",
}

# Initialize an empty DatasetDict
dataset_dict = DatasetDict()

# Get the list of dataset names from the directory names
dataset_names = [
    name for name in os.listdir("data") if os.path.isdir(os.path.join("data", name))
]

# Iterate over the dataset names
for dataset_name in tqdm(dataset_names, desc="Datasets"):
    # Get the list of split names from the directory names
    split_names = [
        name
        for name in os.listdir(os.path.join("data", dataset_name))
        if os.path.isdir(os.path.join("data", dataset_name, name))
    ]

    # Iterate over the split names
    for split_name in tqdm(split_names, desc="Splits", leave=False):
        # Get the list of JSON files in the split directory
        json_files = [
            file
            for file in os.listdir(os.path.join("data", dataset_name, split_name))
            if file.endswith(".jsonl")
        ]

        # Iterate over the JSON files
        for json_file in json_files:
            # Extract the model name from the file name (assuming the format is "model_name.jsonl")
            model_name = os.path.splitext(json_file)[0]

            # Check if the JSON file is empty
            file_path = os.path.join("data", dataset_name, split_name, json_file)
            if os.path.getsize(file_path) > 0:
                print(
                    f"Loading dataset: {dataset_name} - {model_name} - {split_name} at {file_path}"
                )

                df = pd.read_json(file_path, lines=True)
                df["hypotheses_concat"] = df["hypotheses"].apply(
                    lambda x: [item["hypothesis"].strip() for item in x]
                )
                df["hypotheses_normalized_concat"] = df["hypotheses"].apply(
                    lambda x: [item["hypothesis_normalized"].strip() for item in x]
                )
                df["input"] = df["hypotheses"].apply(
                    lambda x: str([item["hypothesis"].strip() for item in x])
                )
                # output is reference
                df["output"] = df["reference"]
                # TODO: depending on the ASR
                df["input_type"] = "Punctuated & Cased"
                df["output_type"] = dataset_to_pnc_type[dataset_name]
                df["instruction"] = generate_instructions(
                    InputType.PUNCTUATED_CASED,
                    OutputType(dataset_to_pnc_type[dataset_name]),
                )
                # Load the dataset
                # dataset = load_dataset("json", data_files=file_path, split="train")

                dataset = Dataset.from_pandas(df)

                # Add the dataset to the DatasetDict with the desired format
                dataset_dict[f"{dataset_name}"] = dataset_dict.get(
                    f"{dataset_name}", {}
                )
                dataset_dict[f"{dataset_name}"][f"{model_name}"] = dataset_dict[
                    f"{dataset_name}"
                ].get(f"{model_name}", {})

                # Combine the 'train' splits
                if split_name.startswith("train"):
                    if "train" not in dataset_dict[f"{dataset_name}"][f"{model_name}"]:
                        dataset_dict[f"{dataset_name}"][f"{model_name}"]["train"] = (
                            dataset
                        )
                    else:
                        dataset_dict[f"{dataset_name}"][f"{model_name}"]["train"] = (
                            concatenate_datasets(
                                [
                                    dataset_dict[f"{dataset_name}"][f"{model_name}"][
                                        "train"
                                    ],
                                    dataset,
                                ]
                            )
                        )
                        # dataset_dict[f"{dataset_name}"][f"{model_name}"]['train'] = Dataset.
                else:
                    dataset_dict[f"{dataset_name}"][f"{model_name}"][
                        f"{split_name}"
                    ] = dataset
            else:
                print(f"Skipping empty file: {file_path}")
    pprint(dataset_dict)

    for model_name, splits in dataset_dict[dataset_name].items():
        print(f"Pushing dataset: {dataset_name} - {model_name}")
        DatasetDict(
            {split_name: dataset for split_name, dataset in splits.items()}
        ).push_to_hub(
            "yentinglin/asr_correction_datasets",
            config_name=f"{dataset_name}-{model_name}",
        )
