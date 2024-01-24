import inflect
import numpy as np
import torch
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import Tracker

LABLES = ["1", "2", "3", "4"]
OBO_EQUIV_LABELS = ["Yes", "yes", "YES"]

INFLECT_ENGINE = inflect.engine()


def update_labels(args):
    global LABLES
    if args.dataset.startswith("iq50"):
        LABLES += ["5", "6"]
    elif args.dataset.startswith("raven-10k"):
        LABLES += ["5", "6", "7", "8"]


def get_torch_dtype(args):
    if args.fp16 and args.bf16:
        raise ValueError
    if args.fp16:
        return torch.float16
    elif args.bf16:
        return torch.bfloat16
    else:
        return torch.float32


def get_prompt(args, sample):
    if args.sequential:
        _shape = "sequence"
        _position = "last"
    else:
        _shape = "matrix"
        _position = "bottom right"
    if "raven" in args.dataset:
        _num = "eight"
        _labels = "1, 2, 3, 4, 5, 6, 7, or 8"
    elif "iq50" in args.dataset:
        _num = "six"
        _labels = "1, 2, 3, 4, 5, or 6"
    elif "marvel" in args.dataset:
        _num = "four"
        _labels = "1, 2, 3, or 4"
    else:
        raise ValueError

    if args.mode == "text-description":
        prompt = sample["text_description"] + "\n\n"
    else:
        prompt = ""

    if "consistency" in args.mode:
        prompt += sample["consistency_question"]
    else:
        prompt += (
            "You are given a puzzle. "
            f"The puzzle features a set of visual patterns arranged in a {_shape} on the top, with the {_position} piece missing, and {_num} options at the bottom (marked by {_labels}). "
            f"Which option (either {_labels}) fills the missing piece best?"
        )

    if args.mode == "text-description":
        prompt = prompt.replace(" visual ", " ")

    if args.mode == "hints-general":
        prompt += " Hint: Focus on the row-wise and column-wise changes regarding color, orientation, and shape of the puzzle pieces."
    elif args.mode == "hints-sample":
        prompt += f" Hint: {sample['hint'].capitalize()}."

    if args.mode in [
        "zs-cot",
        "hints-general",
        "hints-sample",
        "hints-corrective",
        "text-description",
    ]:
        prompt += " Let's think step by step."
    elif args.mode == "icl-cot-asym" and (
        args.model.startswith("instructblip")
        or args.model.startswith("llava")
        or args.model.startswith("bakLlava")
        or args.model.startswith("MMICL")
    ):
        for i, icl_example in enumerate(sample["icl_examples"], 1):
            prompt += f"\n\nDemonstration {i}:\n\n{icl_example['text_description']}\n\n{get_icl_text(args, icl_example)}"
        prompt += "\n\nLet's solve the puzzle in the image, step by step, similar to the demonstrations."
    elif args.mode in "one-by-one" or "icl" in args.mode:
        raise NotImplementedError
    return prompt


def generate(args, model, inputs, **kwargs):
    return model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=args.max_generation_length,
        min_new_tokens=1,
        **kwargs,
    )


def generate_scores(*args, **kwargs):
    return generate(
        *args,
        return_dict_in_generate=True,
        output_scores=True,
        renormalize_logits=True,
        **kwargs,
    )


def get_label(args, outputs, processor, labels):
    max_p, predicted_label = -1, None
    for scores in outputs.scores:
        for label in labels:
            idx = processor(label)
            p = np.exp(scores[0][idx].item())
            if p > max_p:
                max_p = p
                predicted_label = label
        if args.threshold is not None and max_p > args.threshold:
            break
    return max_p, predicted_label


def run(args, device, _get_model, _get_prompt, _generate):
    model, processor = None, None
    if _get_model is not None:
        model, processor = _get_model(args, device)
    if args.stats:
        print(f"{args.model} #params: {model.num_parameters()}")
        return
    dataset, n = get_dataset(args)

    tracker = Tracker(args)
    with tqdm(total=n, leave=False) as pbar:
        for i, sample in enumerate(dataset, 1):
            if tracker.is_cached(i):
                print(f"skipping question {i}: cached.")
                pbar.update()
                continue

            if args.mode == "hints-corrective" and sample["corrective_hint"] is None:
                print(f"skipping question {i}: no corrective hint.")
                tracker.update(i, sample["output"], sample["gold_label"], -1)
                pbar.update()
                continue

            if "consistency" in args.mode and sample["consistency_question"] is None:
                print(f"skipping question {i}: no consistency question.")
                pbar.update()
                continue

            for _ in range(args.num_retries):
                images = sample["images"]
                sample["images"] = (
                    images[: -len(LABLES)] if args.mode == "one-by-one" else images
                )
                prompt = _get_prompt(args, sample)
                p, response = _generate(args, model, processor, images, prompt, device)
                if response is not None:
                    break
            else:
                print(f"skipping question {i}: max retries exceeded.")

            tracker.update(i, response, sample["gold_label"], p)

            pbar.update()
            pbar.set_postfix(acc=tracker.accuracy())

            if args.debug:
                print(f"Question {i}")
                print(f"Prompt: {prompt}")
                print(f"Gold Label: {sample['gold_label']}")
                print(f"Response: {response}")
                exit()

    print(f"{args.model},{args.dataset},{args.mode}: acc={tracker.accuracy()}")


def get_icl_prompt():
    return (
        "You are given a puzzle. "
        "The puzzle features a set of visual patterns arranged in a matrix on the top, with the bottom right piece missing, and a set of options at the bottom. "
        "Which option fills the missing piece best?"
    )


def get_icl_text(args, icl_example):
    return (
        icl_example["cot_reasoning"]
        if "cot" in args.mode
        else f"The answer is {icl_example['gold_label']}."
    )
