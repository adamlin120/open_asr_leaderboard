import os
import ipdb
import json
import logging
import argparse
import evaluate
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

# run "export PYTHONPATH="..":$PYTHONPATH" before import normalizer
os.environ["PYTHONPATH"] = "..:" + os.environ.get("PYTHONPATH", "")
from normalizer.data_utils import normalizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech correction model")
    parser.add_argument(
        "--results_file",
        default="neko_eval_results.json",
        help="Output file for evaluation results",
    )
    parser.add_argument(
        "--asr_dataset_name",
        default="yentinglin/asr_correction_datasets",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--asr_dataset_config_name", required=True, help="Name of the dataset config"
    )
    parser.add_argument(
        "--asr_dataset_split_name", default="test", help="Name of the dataset split"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the speech correction model",
    )
    args = parser.parse_args()

    dataset_name = args.asr_dataset_name
    model_name = args.model_name
    config_name = args.asr_dataset_config_name
    results_file = args.results_file

    logger.info(f"Loading dataset {dataset_name} with config {config_name}")
    dataset = load_dataset(dataset_name, config_name, split=args.asr_dataset_split_name)

    wer_metric = evaluate.load("wer")

    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=0, max_tokens=128, stop=['<|eot_id|>'])
    llm = LLM(model=model_name)

    prompts = [
        tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "user",
                    "content": f"{example['instruction']}\n{example['input']}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in dataset
    ]

    outputs = llm.generate(prompts, sampling_params, )

    outputs.sort(key=lambda x: int(x.request_id))

    assert len(outputs) == len(
        dataset
    ), f"Expected {len(dataset)} outputs, got {len(outputs)}"

    references = []
    predictions = []

    # Print the outputs.
    for output, example in zip(outputs, dataset):
        # prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        predictions.append(normalizer(generated_text))
        references.append(normalizer(example["output"]))

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    print(
        f"WER (normalized) after correction: {wer}%, Dataset: {dataset_name}, Config: {config_name}"
    )
    # ipdb.set_trace()

    top1_hypotheses = [normalizer(example['hypotheses_concat'][-1]) for example in dataset]
    # calculate wer before correction
    wer_before = wer_metric.compute(references=references, predictions=top1_hypotheses)
    wer_before = round(100 * wer_before, 2)
    print(
        f"WER before correction: {wer_before}%, Dataset: {dataset_name}, Config: {config_name}"
    )

    # calculate relative improvement percentage
    improvement = (wer_before - wer) / wer_before * 100
    print(f"Relative improvement: {improvement}%")

    result = {
        "dataset": dataset_name,
        "config": config_name,
        "wer_before_correction": wer_before,
        "wer_after_correction": wer,
        "relative_improvement": improvement,
    }
    results.append(result)

    with open(results_file, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()
