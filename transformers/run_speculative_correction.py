import argparse
import os
import re
import json
import torch
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, set_seed
import evaluate
from datasets import load_dataset, Audio
from tqdm import trange
from normalizer import EnglishTextNormalizer
from normalizer.normalizer import remove_symbols_and_diacritics
from normalizer.data_utils import get_text, is_target_text_in_range, load_data, write_manifest

set_seed(42)


class EnglishTextCaptalizedNormalizer(EnglishTextNormalizer):
    def __call__(self, s: str):
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep some symbols for numerics

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s


wer_metric = evaluate.load("wer")


def dataset_iterator(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args):
    device = args.device
    model_id = args.model_id
    is_multilingual = not model_id.endswith("en")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    dataset = load_data(args)

    # normalizer = EnglishTextCaptalizedNormalizer()
    normalizer = EnglishTextNormalizer()

    def normalize(batch):
        batch["original_text"] = get_text(batch)
        batch["norm_text"] = normalizer(batch["original_text"])
        return batch


    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalize)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    def inference(
        batch,
        num_return_sequences: int = 5,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        # 1. Pre-process the audio data to log-mel spectrogram inputs
        audio = [sample["array"] for sample in batch["audio"]]
        input_features = processor(
            audio, sampling_rate=batch["audio"][0]["sampling_rate"], return_tensors="pt"
        ).input_features
        input_features = input_features.to(device, dtype=torch_dtype)

        # 2. Auto-regressively generate the predicted token ids
        pred_ids = model.generate(
            input_features,
            max_new_tokens=128,
            # do_sample=do_sample,
            # temperature=temperature,
            num_return_sequences=num_return_sequences,
            num_beams=5, num_beam_groups=5, diversity_penalty=1.0,
            language='english' if is_multilingual else None,
            task="transcribe" if is_multilingual else None
        )
        trans = processor.batch_decode(pred_ids, skip_special_tokens=True)
        # len of trans is now num_return_sequences * batch_size
        transcription = trans[::num_return_sequences]
        # transcription to be list of list
        hypotheses = [
            trans[i * num_return_sequences : (i + 1) * num_return_sequences]
            for i in range(len(audio))
        ]

        # 3. Decode the token ids to the final transcription
        batch["transcription"] = transcription
        batch["hypotheses"] = hypotheses
        return batch

    predictions = []
    hypotheses = []
    references = []

    file_name = f"data/{args.dataset}/{args.split}/{args.model_id.split('/')[-1]}.jsonl"
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = open(file_name, "w")

    # run streamed inference
    for i, start in enumerate(trange(0, len(dataset), args.batch_size)):
        # yield {**item["audio"], "reference": item["norm_text"]}
        item = dataset[start : start + args.batch_size]
        prediction = inference(item)
        predictions.extend(
            [normalizer(x) for x in prediction["transcription"]]
        )
        hypotheses.extend(prediction["hypotheses"])
        references.extend(item["norm_text"])
        # print hypothesis and reference
        for hyp, ref, text in zip(
            prediction["hypotheses"], item["norm_text"], item["text"]
        ):
            # print(f"Reference: {text}")
            # print(f"Reference normalized: {ref}")
            hypothesis_data = {
                "reference": text,
                "reference_normalized": ref,
                "hypotheses": []
            }
            for i, h in enumerate(hyp):
                # print(f"Hypothesis {i}: {h} \t {normalizer(h)}")
                hypothesis_data["hypotheses"].append({
                    "hypothesis_index": i,
                    "hypothesis": h,
                    "hypothesis_normalized": normalizer(h)
                })
            jsonl_file.write(json.dumps(hypothesis_data) + "\n")
            # print()
    jsonl_file.close()

    # Write manifest results
    manifest_path = write_manifest(
        references,
        predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="open-asr-leaderboard/datasets-test-only",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
