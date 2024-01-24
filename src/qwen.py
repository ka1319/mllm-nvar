import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (
    INFLECT_ENGINE,
    LABLES,
    OBO_EQUIV_LABELS,
    generate,
    generate_scores,
    get_icl_prompt,
    get_icl_text,
    get_label,
    get_prompt,
    get_torch_dtype,
    run,
)


def _get_model(args, device):
    model_name = f"Qwen/{args.model}"
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    torch_dtype = get_torch_dtype(args)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(args),
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        fp32=(torch_dtype == torch.float32),
        trust_remote_code=True,
    ).to(device)
    return model, processor


def _get_prompt(args, sample, query_images=None, answer_image=None):
    if args.mode == "one-by-one":
        if query_images is None or answer_image is None:
            return None
        num_query_images = INFLECT_ENGINE.number_to_words(len(query_images))
        prompt = [
            {"text": f"Here are {num_query_images} images: "},
            *[{"image": image} for image in query_images],
            {"text": "The following image is: "},
            {"image": answer_image},
            {"text": "Is it correct?"},
        ]
    elif "icl" in args.mode:
        prompt = [
            {"text": get_icl_prompt()},
        ]
        for i, icl_example in enumerate(sample["icl_examples"], 1):
            if "asym" in args.mode:
                prompt += [
                    {
                        "text": f"\n\nDemonstration {i}:\n\n{icl_example['text_description']}",
                    },
                ]
            else:
                prompt += [{"image": image} for image in icl_example["images"]]
            prompt += [
                {
                    "text": "\n\n" + get_icl_text(args, icl_example),
                },
            ]
        prompt += [
            *[{"image": image} for image in sample["images"]],
        ]
        if "asym" in args.mode:
            prompt += [
                {
                    "text": "\n\nLet's solve the puzzle in the image, step by step, similar to the demonstrations.",
                },
            ]
    elif args.mode == "text-description":
        prompt = [
            {"text": get_prompt(args, sample)},
        ]
    else:
        prompt = [
            *[{"image": image} for image in sample["images"]],
            {"text": get_prompt(args, sample)},
        ]
    return prompt


def _generate(args, model, processor, _, prompt, device):
    query = processor.from_list_format(prompt)
    inputs = processor(query, return_tensors="pt").to(device)
    if args.mode in [
        "zs-cot",
        "icl-cot",
        "icl-cot-asym",
        "consistency-easy",
        "consistency-hard",
        "text-description",
    ]:
        outputs = generate(args, model, inputs)
        return -1, processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    else:
        outputs = generate_scores(args, model, inputs)

        _processor = lambda x: processor(
            x, return_tensors="pt", add_special_tokens=False
        ).input_ids[0][-1]
        _labels = OBO_EQUIV_LABELS if args.mode == "one-by-one" else LABLES
        p, label = get_label(args, outputs, _processor, _labels)
        return p, label


def _generate_obo(args, model, processor, images, _, device):
    query_images, answer_images = images[: -len(LABLES)], images[-len(LABLES) :]
    max_p, predicted_label = -1, None
    for answer_image, label in zip(answer_images, LABLES):
        _prompt = _get_prompt(args, None, query_images, answer_image)
        p, _ = _generate(args, model, processor, _, _prompt, device)
        if p > max_p:
            max_p = p
            predicted_label = label
    return max_p, predicted_label


def qwen(args, device):
    func = _generate_obo if args.mode == "one-by-one" else _generate
    return run(args, device, _get_model, _get_prompt, func)
