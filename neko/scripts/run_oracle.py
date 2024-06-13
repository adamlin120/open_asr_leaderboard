from datasets import load_dataset
from typing import List
import evaluate
import numpy as np

wer_metric = evaluate.load("wer")


def calculate_nbest_oracle_wer(hypotheses: List[str], oracle_transcript: str) -> float:
    wer = min(
        wer_metric.compute(references=[oracle_transcript], predictions=[hyp])
        for hyp in hypotheses
    )
    wer = round(100 * wer, 2)
    return wer


def calculate_compositional_oracle_wer(
    hypotheses: List[str], oracle_transcript: str
) -> float:
    """
    Ref: https://proceedings.neurips.cc/paper_files/paper/2023/file/6492267465a7ac507be1f9fd1174e78d-Paper-Datasets_and_Benchmarks.pdf
    Calculate the Compositional Oracle

    Args:
        hypotheses (List[str]): The list of input sequences.
        oracle_transcript (str): The ground-truth output sequence.

    Returns:
        float: The token-level oracle value.
    """
    # 1. Is each token in ground-truth included in 1-best or 2-to-n-best sequence(s)?
    match = [0] * 4
    gt_tokens = oracle_transcript.split()

    for gt_token in gt_tokens:
        in_best_1 = gt_token in hypotheses[0]
        in_best_2_n = any(gt_token in best_2_n_hypo for best_2_n_hypo in hypotheses[1:])

        if in_best_1 and not in_best_2_n:
            match[0] += 1
        elif in_best_1 and in_best_2_n:
            match[1] += 1
        elif not in_best_1 and in_best_2_n:
            match[2] += 1
        else:
            match[3] += 1

    sum_match = sum(match)
    match = [round(100 * count / sum_match, 2) for count in match]

    return match[3]


def main(
    dataset_name="ami",
    asr_name="distil-large-v2",
):
    subset = f"{dataset_name}-{asr_name}"
    dataset = load_dataset("yentinglin/asr_correction_datasets", subset, split="test")

    """
    >> dataset
    Dataset({
        features: ['reference', 'reference_normalized', 'hypotheses', 'hypotheses_concat', 'hypotheses_normalized_concat', 'input', 'output', 'input_type', 'output_type', 'instruction'],
        num_rows: 1842
    })
    """

    avg_onb = np.average(
        dataset.map(
            lambda x: {
                "onb": calculate_nbest_oracle_wer(
                    x["hypotheses_concat"], x["reference"]
                )
            }
        )["onb"]
    )

    avg_ocp = np.average(
        dataset.map(
            lambda x: {
                "ocp": calculate_compositional_oracle_wer(
                    x["hypotheses_concat"], x["reference"]
                )
            }
        )["ocp"]
    )

    # Calculate oracle WER for each example in the dataset
    print(f"Dataset: {dataset_name}")
    print(f"ASR: {asr_name}")
    print(f"Number of examples: {len(dataset)}")
    print(f"Average N-best oracle WER (onb): {avg_onb:.2f}")
    print(f"Average Compositional oracle WER (ocp): {avg_ocp:.2f}")


if __name__ == "__main__":
    main()
