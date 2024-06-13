import argparse
import loggign
import os
import shutil
import torch
import evaluate
import soundfile
from tqdm import tqdm
from normalizer import data_utils
from nemo.collections.asr.models import EncDecMultiTaskModel
from datasets import load_dataset
from nemo.collections.asr.models import ASRModel
from transformers import pipeline

DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")

wer_metric = evaluate.load("wer")

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
INSTRUCTION = (
    "The following text contains 5-best hypotheses from an Automatic Speech Recognition system. "
    "As part of a speech recognition task, please perform error correction on the hypotheses to generate the most accurate transcription of the spoken text. "
    "The input hypotheses include both punctuation and proper case information. "
    "Your output transcription should include both punctuation and proper case information."
)


def dataset_iterator(dataset):
    for i, item in enumerate(dataset):
        yield {
            **item["audio"],
            "reference": item["norm_text"],
            "audio_filename": f"file_{i}",
            "sample_rate": 16_000,
            "sample_id": i,
        }


def write_audio(buffer, cache_prefix) -> list:
    cache_dir = os.path.join(DATA_CACHE_DIR, cache_prefix)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir)

    data_paths = []
    for idx, data in enumerate(buffer):
        fn = os.path.basename(data["audio_filename"])
        fn = os.path.splitext(fn)[0]
        path = os.path.join(cache_dir, f"{idx}_{fn}.wav")
        data_paths.append(path)

        soundfile.write(path, data["array"], samplerate=data["sample_rate"])

    return data_paths


def pack_results(results: list, buffer, transcriptions):
    for sample, transcript in zip(buffer, transcriptions):
        result = {
            "reference": sample["reference"],
            "pred_text": transcript[0].text,
            "hypotheses": [h.text for h in transcript],
        }

        results.append(result)
    return results


def buffer_audio_and_transcribe(
    model: ASRModel, dataset, batch_size: int, cache_prefix: str, verbose: bool = True
):
    buffer = []
    results = []
    for sample in tqdm(
        dataset_iterator(dataset),
        desc="Evaluating: Sample id",
        unit="",
        disable=not verbose,
    ):
        buffer.append(sample)

        if len(buffer) == batch_size:
            filepaths = write_audio(buffer, cache_prefix)
            try:
                transcriptions = model.transcribe(
                    filepaths,
                    return_hypotheses=True,
                    batch_size=32,
                    channel_selector="average",
                    verbose=False,
                )[1]
            except Exception as e:
                print(e)
                # if transcriptions fail, print error and continue
                print(f"Transcription failed for batch: {filepaths}")
                continue
            # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
            if type(transcriptions) == tuple and len(transcriptions) == 2:
                transcriptions = transcriptions[0]
            results = pack_results(results, buffer, transcriptions)
            buffer.clear()

    if len(buffer) > 0:
        filepaths = write_audio(buffer, cache_prefix)
        try:
            transcriptions = model.transcribe(
                filepaths,
                return_hypotheses=True,
                batch_size=32,
                channel_selector="average",
                verbose=False,
            )[1]
        except Exception as e:
            print(e)
            # if transcriptions fail, print error and continue
            print(f"Transcription failed for batch: {filepaths}")
            return results
        # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
        if type(transcriptions) == tuple and len(transcriptions) == 2:
            transcriptions = transcriptions[0]
        results = pack_results(results, buffer, transcriptions)
        buffer.clear()

    # Delete temp cache dir
    if os.path.exists(DATA_CACHE_DIR):
        shutil.rmtree(DATA_CACHE_DIR)

    return results


def canary_temperature_sampling():
    # install the PR branch
    # !pip install git+https://github.com/NVIDIA/NeMo.git@canary-temp-sampling#egg=nemo_toolkit[asr]
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")

    # load model
    canary_model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")

    # load dataset
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")

    # update dcode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.strategy = "greedy"
    decode_cfg.greedy.temperature = 1.0
    decode_cfg.greedy.n_samples = 5
    decode_cfg.greedy.max_generation_delta = decode_cfg.beam.max_generation_delta
    canary_model.change_decoding_strategy(decode_cfg)

    hyps = canary_model.transcribe(audio=[dataset["validation"][0]["file"]])[1]

    print(["".join([char.strip() or " " for char in hyp]) for hyp in hyps])

    [
        "<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|> mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.<|endoftext|><pad><pad>",
        "<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|> mister Quilter is the Apostle of the Middle Classes, and we are glad to welcome his Gospel.<|endoftext|>",
        "<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|> mister Quilter is the Apostle of the middle classes, and we are glad to welcome his gospel.<|endoftext|><pad>",
        "<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|> mister Quilter is the apostle of the middle classes, and we are glad to welcome His gospel.<|endoftext|><pad>",
        "<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|> mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.<|endoftext|><pad><pad>",
    ]


def main(args):
    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    if args.model_id.endswith(".nemo"):
        asr_model = ASRModel.restore_from(args.model_id, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(args.model_id, map_location=device)  # type: ASRModel
    if "canary" in args.model_id:
        # update dcode params
        canary_decode_cfg = asr_model.cfg.decoding
        canary_decode_cfg.beam.beam_size = 5
        canary_decode_cfg.beam.return_best_hypothesis = False
        asr_model.change_decoding_strategy(canary_decode_cfg)
    elif "parakeet" in args.model_id:
        decode_cfg = asr_model.cfg.decoding
        decode_cfg.beam.beam_size = 5
        decode_cfg.beam.return_best_hypothesis = False
        decode_cfg.strategy = "beam"
        asr_model.change_decoding_strategy(decode_cfg)
    asr_model.freeze()

    correction_pipe = pipeline(
        "text-generation",
        model="yentinglin/speech-correction-lm-asr-all",
        device=args.device,
    )

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    predictions = []
    references = []

    # run streamed inference
    cache_prefix = (
        f"{args.model_id.replace('/', '-')}-{args.dataset_path.replace('/', '')}-"
        f"{args.dataset.replace('/', '-')}-{args.split}"
    )
    results = buffer_audio_and_transcribe(
        asr_model, dataset, args.batch_size, cache_prefix, verbose=True
    )
    prompts = [
        correction_pipe.tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "user",
                    "content": ALPACA_TEMPLATE.format(
                        instruction=INSTRUCTION, input=str(result["hypotheses"])
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for result in results
    ]
    for i, x in enumerate(
        correction_pipe(
            prompts,
            max_length=1500,
            return_full_text=False,
            do_sample=False,
            batch_size=8,
        )
    ):
        results[i]["pred_after_correction"] = x[0]["generated_text"].strip()
    for sample in results:
        print("Reference:", sample["reference"])
        print("Prediction:", data_utils.normalizer(sample["pred_text"]))
        print(
            "Prediction after correction:",
            data_utils.normalizer(sample["pred_after_correction"]),
        )
        print()

        predictions.append(data_utils.normalizer(sample["pred_after_correction"]))
        # predictions.append(data_utils.normalizer(sample["pred_text"]))
        references.append(sample["reference"])

    # Write manifest results
    manifest_path = data_utils.write_manifest(
        references,
        predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        hypothesis=[sample["hypotheses"] for sample in results],
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
        default="nvidia/canary-1b",
        help="Model identifier. Should be loadable with NVIDIA NeMo.",
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
        default=-1,
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
    parser.set_defaults(streaming=True)

    main(args)
