import argparse
import json
import os


class Tracker(object):
    def __init__(self, args):
        self._fn = args.log_file
        self._results = self._load(args.rewrite)

    def _load(self, rewrite):
        results = {}
        if not rewrite and os.path.exists(self._fn):
            with open(self._fn, "r") as f:
                for l in f:
                    results.update(json.loads(l))
        return results

    def _save(self):
        with open(self._fn, "w") as f:
            for key in sorted(self._results, key=lambda x: int(x.split("-")[-1])):
                f.write(json.dumps({key: self._results[key]}) + "\n")

    def _key(self, idx):
        return f"exp-{idx}"

    def update(self, idx, predicted, gold, p):
        key = self._key(idx)
        self._results[key] = {"predicted": predicted, "gold": gold, "p": p}
        self._save()

    def accuracy(self):
        correct = 0
        for result in self._results.values():
            if result["predicted"] == result["gold"]:
                correct += 1
        return correct / len(self._results)

    def is_cached(self, idx):
        key = self._key(idx)
        return key in self._results


def get_log_file(args):
    _args = [
        "matrix" if not args.sequential else "sequential",
    ]
    if args.threshold is not None:
        _args.append(f"threshold={args.threshold}")
    if "icl" in args.mode:
        _args += [
            f"icl_aux_dataset={args.icl_aux_dataset}",
            f"k={args.k}",
        ]
    if args.run_id is not None:
        _args.append(f"run_id={args.run_id}")
    _args = "-".join(_args)
    log_file = f"logs/{args.dataset}-{args.mode}-{args.model}-{_args}.jsonl"
    return log_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            # closed-source
            "gpt-4-vision-preview",
            "gemini-pro-vision",
            # instruction-tuned
            "instructblip-vicuna-7b",
            "instructblip-vicuna-13b",
            "instructblip-flan-t5-xl",
            "instructblip-flan-t5-xxl",
            "idefics-9b-instruct",
            "idefics-80b-instruct",
            "llava-1.5-7b-hf",
            "llava-1.5-13b-hf",
            "bakLlava-v1-hf",
            "MMICL-vicuna-7b",
            "MMICL-vicuna-13b",
            "MMICL-Instructblip-T5-xl",
            "MMICL-Instructblip-T5-xxl",
            "Qwen-VL-Chat",
            # pre-trained
            "blip2-opt-2.7b",
            "blip2-opt-6.7b",
            "blip2-flan-t5-xl",
            "blip2-flan-t5-xxl",
            "idefics-9b",
            "idefics-80b",
            "fuyu-8b",
            "Qwen-VL",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "iq50",
            "raven-10k",
            "raven-10k-sampled",
            "marvel",
        ],
        default="iq50",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "label",
            "zs-cot",
            "one-by-one",
            "text-description",
            "hints-general",
            "hints-sample",
            "hints-corrective",
            "consistency-easy",
            "consistency-hard",
            "icl",
            "icl-cot",
            "icl-cot-asym",
        ],
        default="label",
    )
    parser.add_argument(
        "--icl-aux-dataset",
        type=str,
        choices=[
            "iq50",
            "raven-10k-sampled",
            "marvel",
        ],
        default="iq50",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base64",
        action="store_true",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
    )
    parser.add_argument(
        "--max_generation_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--stats",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.log_file is None:
        args.log_file = get_log_file(args)

    return args
