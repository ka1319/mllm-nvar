import base64
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

ICL_NUM_SAMPLES = {
    "iq50": 50,
    "raven-10k": 70000,
    "raven-10k-sampled": 3500,
    "marvel": 175,
}

ICL_GOLD_LABELS = None
ICL_TEXT_DESCRIPTIONS = None
ICL_COT_REASONINGS = None


def _load_image(fn, args):
    if args.base64:
        with open(fn, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if args.raw:
        return fn
    return Image.open(fn).convert("RGB")


def _read_csv(fn):
    if not os.path.exists(fn):
        return []
    df = pd.read_csv(fn, dtype=str, header=None)
    return df.values[:, 0].tolist()


def _read_json(fn):
    if not os.path.exists(fn):
        return {}
    with open(fn, "r") as f:
        return json.load(f)


def _read_jsonl(fn, key):
    if not os.path.exists(fn):
        return {}
    data = {}
    with open(fn, "r") as f:
        for line in f:
            k, v = list(json.loads(line).items())[0]
            data[k] = v[key]
    return data


def _get_gold_labels(dataset):
    return _read_csv(f"data/{dataset}/labels.tsv")


def _get_text_descriptions(dataset):
    return _read_csv(f"data/{dataset}/descriptions.tsv")


def _get_cot_reasonings(dataset):
    return _read_csv(f"data/{dataset}/cot_reasonings.tsv")


def _get_hints(dataset):
    return _read_csv(f"data/{dataset}/hints.tsv")


def _get_corrective_hints(dataset, model):
    return _read_json(f"data/{dataset}/{model}-corrective-reasonings.json")


def _get_outputs(dataset, model):
    return _read_jsonl(f"outputs/{dataset}-zs-cot-{model}-matrix.jsonl", "predicted")


def _get_consistency_questions(dataset, mode):
    return _read_jsonl(f"data/{dataset}/consistency_questions_{mode}.jsonl", "question")


def _get_icl_examples(args, i):
    icl_examples = []

    global ICL_GOLD_LABELS, ICL_TEXT_DESCRIPTIONS, ICL_COT_REASONINGS
    if ICL_GOLD_LABELS is None:
        ICL_GOLD_LABELS = _get_gold_labels(args.icl_aux_dataset)
    if ICL_TEXT_DESCRIPTIONS is None:
        ICL_TEXT_DESCRIPTIONS = _get_text_descriptions(args.icl_aux_dataset)
    if ICL_COT_REASONINGS is None:
        ICL_COT_REASONINGS = _get_cot_reasonings(args.icl_aux_dataset)

    indices = np.setdiff1d(np.arange(1, ICL_NUM_SAMPLES[args.icl_aux_dataset] + 1), [i])
    examples_indices = np.random.choice(indices, args.k, replace=False)
    for example_idx in examples_indices:
        mode = "sequential" if args.sequential else "matrix"
        fn = glob(f"data/{args.icl_aux_dataset}/{mode}/*-{example_idx}.png")[0]
        images = [
            _load_image(fn, args),
        ]
        text_description = (
            ICL_TEXT_DESCRIPTIONS[example_idx - 1]
            if len(ICL_TEXT_DESCRIPTIONS) > 0
            else None
        )
        cot_reasoning = (
            ICL_COT_REASONINGS[example_idx - 1] if len(ICL_COT_REASONINGS) > 0 else None
        )
        gold_label = ICL_GOLD_LABELS[example_idx - 1]

        icl_example = {
            "images": images,
            "text_description": text_description,
            "cot_reasoning": cot_reasoning,
            "gold_label": gold_label,
        }
        icl_examples.append(icl_example)

    return icl_examples


def load(args):
    gold_labels = _get_gold_labels(args.dataset)
    text_descriptions = _get_text_descriptions(args.dataset)
    hints = _get_hints(args.dataset)
    corrective_hints = _get_corrective_hints(args.dataset, args.model)
    outputs = _get_outputs(args.dataset, args.model)
    consistency_questions = _get_consistency_questions(
        args.dataset, args.mode.split("-")[-1]
    )

    n = len(gold_labels)
    for i, gold_label in zip(range(1, n + 1), gold_labels):
        if args.mode == "one-by-one":
            query_fns = sorted(glob(f"data/{args.dataset}/obo/*-{i}/query-*.png"))
            answer_fns = sorted(glob(f"data/{args.dataset}/obo/*-{i}/answer-*.png"))
            query_images = [_load_image(fn, args) for fn in query_fns]
            answer_images = [_load_image(fn, args) for fn in answer_fns]
            images = query_images + answer_images
        else:
            mode = "sequential" if args.sequential else "matrix"
            fn = glob(f"data/{args.dataset}/{mode}/*-{i}.png")[0]
            images = [
                _load_image(fn, args),
            ]
        text_description = (
            text_descriptions[i - 1] if len(text_descriptions) > 0 else None
        )
        hint = hints[i - 1] if len(hints) > 0 else None
        corrective_hint = corrective_hints.get(f"exp-{i}", None)
        output = outputs.get(f"exp-{i}", None)
        icl_examples = _get_icl_examples(args, i) if "icl" in args.mode else None
        consistency_question = consistency_questions.get(f"exp-{i}", None)
        yield {
            "images": images,
            "text_description": text_description,
            "hint": hint,
            "corrective_hint": corrective_hint,
            "output": output,
            "icl_examples": icl_examples,
            "consistency_question": consistency_question,
            "gold_label": gold_label,
        }


def get_dataset(args):
    return load(args), len(_get_gold_labels(args.dataset))
