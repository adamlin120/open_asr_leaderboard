from datasets import load_dataset
from prompt_utils import generate_instructions, InputType, OutputType

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("yentinglin/canary_training_set", 'parakeet')
dataset = dataset.map(
    lambda x: {
        "input_type": "Normalized",
        "output_type": "Normalized",
        "instruction": generate_instructions(InputType.NORMALIZED, OutputType.NORMALIZED),
    }
)
dataset.push_to_hub("yentinglin/canary_training_set", "parakeet")


dataset = load_dataset("yentinglin/canary_training_set", 'canary')
dataset = dataset.map(
    lambda x: {
        "input_type": "Punctuated & Cased",
        "output_type": "Normalized",
        "instruction": generate_instructions(InputType.NORMALIZED, OutputType.NORMALIZED),
    }
)
dataset.push_to_hub("yentinglin/canary_training_set", "canary")

